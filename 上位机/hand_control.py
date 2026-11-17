"""RAPID Hand 风格的关节映射与飞特 HLS 上位机。"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parent
TICKS_PER_DEGREE = 4096 / 360
# HLS absolute multi-turn registers use a 15-bit magnitude and bit 15 sign.
# These are protocol bounds; each motor's calibrated soft bounds still apply.
MULTITURN_MIN_TICK = -32767
MULTITURN_MAX_TICK = 32767
HOST = "127.0.0.1"
MAX_FRAME_BYTES = 8192
FRAME_TIMEOUT = 0.5

# HLS 内存表。双字节低位在前，反馈的符号位为 bit 15。
MODEL = 3
SERVO_ADDRESS = 5
HLS_BAUD_RATES = (1000000, 500000, 250000, 128000, 115200, 76800, 57600, 38400)
ANGLE_LIMITS = 9
RESOLUTION = 30
MODE = 33
TORQUE_ENABLE = 40
ACCELERATION = 41
GOAL_POSITION = 42
CURRENT_LIMIT = 44
SPEED_LIMIT = 46
TORQUE_LIMIT = 48
PRESENT_POSITION = 56


def number(value, low, high, name):
    if type(value) not in (int, float) or not low <= value <= high:
        raise ValueError(f"{name} 必须在 {low} 到 {high} 之间")
    return value


def integer(value, low, high, name):
    if type(value) is not int:
        raise ValueError(f"{name} 必须是整数")
    return number(value, low, high, name)


@dataclass(frozen=True)
class Motor:
    id: int
    zero_tick: int
    min_tick: int
    max_tick: int
    weights: dict
    current_ma: float
    speed_rpm: float
    allow_multiturn: bool = False


def initial_angles(config):
    """Initial target preview only; does not query or command a device."""
    return {name: max(low, min(high, 0)) for name, (low, high) in config.joints.items()}


@dataclass(frozen=True)
class DifferentialPair:
    """两个关节自由度由两台舵机共同驱动；系数仍以 Motor.weights 为准。"""
    name: str
    flex_joint: str
    side_joint: str
    motor_ids: tuple


class HandConfig:
    def __init__(self, path):
        self._load(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    @classmethod
    def from_data(cls, data):
        config = cls.__new__(cls)
        config._load(data)
        return config

    def to_data(self):
        return {"calibrated": self.calibrated, "model_number": self.model_number,
                "allow_unbound_joints": self.allow_unbound_joints,
                "joint_limits_deg": {name: list(bounds) for name, bounds in self.joints.items()},
                "motors": [asdict(motor) for motor in self.motors.values()]}

    def _load(self, data):
        self.calibrated = data["calibrated"] is True
        self.allow_unbound_joints = data.get("allow_unbound_joints", False)
        if type(self.allow_unbound_joints) is not bool:
            raise ValueError("allow_unbound_joints 必须为布尔值")
        self.model_number = data["model_number"]
        self.joints = data["joint_limits_deg"]
        if not self.joints:
            raise ValueError("至少配置一个关节")
        for name, (low, high) in self.joints.items():
            number(low, -180, 180, name)
            number(high, low, 180, name)

        self.motors = {}
        for entry in data["motors"]:
            motor = Motor(**entry)
            sid = integer(motor.id, 0, 253, "舵机 ID")
            if sid in self.motors:
                raise ValueError(f"重复的舵机 ID {sid}")
            if type(motor.allow_multiturn) is not bool:
                raise ValueError("allow_multiturn 必须为布尔值")
            position_low, position_high = ((MULTITURN_MIN_TICK, MULTITURN_MAX_TICK)
                                           if motor.allow_multiturn else (0, 4095))
            integer(motor.min_tick, position_low, position_high, "min_tick")
            integer(motor.max_tick, motor.min_tick, position_high, "max_tick")
            integer(motor.zero_tick, motor.min_tick, motor.max_tick, "zero_tick")
            number(motor.current_ma, 6.5, 1300, "current_ma")
            number(motor.speed_rpm, 1, 30, "speed_rpm")
            if not motor.weights or motor.weights.keys() - self.joints.keys():
                raise ValueError(f"ID {sid} 的关节映射无效")
            for name, weight in motor.weights.items():
                number(weight, -20, 20, f"ID {sid} / {name}")
                if weight == 0:
                    raise ValueError(f"ID {sid} / {name} 的权重须非零")
            self.motors[sid] = motor
        mapped = {name for motor in self.motors.values() for name in motor.weights}
        if (not self.allow_unbound_joints and mapped != self.joints.keys()) or len(self.motors) > 20:
            raise ValueError("每个关节都须有映射，最多支持 20 台舵机")
        self.differential_pairs = self._find_differential_pairs()

    def _find_differential_pairs(self):
        pairs = {}
        for finger in ("index", "thumb", "middle", "ring", "little"):
            joint = "cmc" if finger == "thumb" else "mcp"
            first = f"{finger}_{joint}_flex"
            second = f"{finger}_{joint}_{'roll' if finger == 'thumb' else 'abd'}"
            if first not in self.joints or second not in self.joints:
                continue
            axes = {first, second}
            motors = [motor for motor in self.motors.values() if axes & motor.weights.keys()]
            name = f"{finger}_{joint}"
            if len(motors) != 2 or any(set(motor.weights) != axes for motor in motors):
                if self.allow_unbound_joints:
                    continue
                raise ValueError(f"{name} 差动组需要两台舵机，且每台都须包含 {first} 和 {second} 的权重")
            a, b = (motors[0].weights[axis] for axis in (first, second))
            c, d = (motors[1].weights[axis] for axis in (first, second))
            if abs(a * d - b * c) <= 1e-9 * max(abs(a), abs(b), abs(c), abs(d)) ** 2:
                raise ValueError(f"{name} 差动矩阵退化，两个自由度无法独立控制；请检查两台舵机的方向和比例")
            pairs[name] = DifferentialPair(name, first, second, tuple(motor.id for motor in motors))
        return pairs

    def describe_mapping(self, angles):
        """计算每个关节对舵机的贡献，供诊断显示；发送必须再通过 targets 的整帧限位检查。"""
        if not isinstance(angles, dict) or angles.keys() != self.joints.keys():
            raise ValueError(f"一帧须包含全部关节，名称为 {', '.join(self.joints)}")
        for name, (low, high) in self.joints.items():
            number(angles[name], low, high, name)
        result = {}
        for sid, motor in self.motors.items():
            contributions = {name: angles[name] * weight for name, weight in motor.weights.items()}
            degrees = sum(contributions.values())
            tick = motor.zero_tick + round(degrees * TICKS_PER_DEGREE)
            result[sid] = {"motor_deg": degrees, "contributions_deg": contributions,
                           "zero_tick": motor.zero_tick, "target_tick": tick,
                           "within_limits": motor.min_tick <= tick <= motor.max_tick}
        return result

    def targets(self, angles):
        targets = {}
        for sid, detail in self.describe_mapping(angles).items():
            motor = self.motors[sid]
            # 任一电机越界即拒绝整帧，不单独截断差动组中的某一台。
            targets[sid] = integer(detail["target_tick"], motor.min_tick, motor.max_tick,
                                   f"ID {sid} 目标位置（差动映射后）")
        return targets


class HlsBus:
    def __init__(self, port_name, baud):
        sdk_path = ROOT / "FTServo_Python"
        if (sdk_path / "scservo_sdk").is_dir() and str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        try:
            from scservo_sdk import COMM_SUCCESS, GroupSyncWrite, PortHandler, hls
        except ImportError as exc:
            raise RuntimeError("请按 README 安装 pyserial 和飞特 FTServo_Python SDK") from exc
        self.success = COMM_SUCCESS
        self.port = PortHandler(port_name)
        self.packet = hls(self.port)
        self.sync = GroupSyncWrite(self.packet, GOAL_POSITION, 2)
        self.enabled_ids = set()
        self.limits = {}
        try:
            if not self.port.setBaudRate(baud):
                raise RuntimeError(f"无法设置串口 {port_name} 的波特率 {baud}")
            self.port.ser.write_timeout = 0.2
        except BaseException:
            if self.port.is_open:
                self.port.closePort()
            raise

    def check(self, result, error, context):
        if result != self.success:
            raise RuntimeError(f"{context}：{self.packet.getTxRxResult(result)}")
        if error:
            raise RuntimeError(f"{context}：舵机状态包错误 0x{error:02X}")

    def read(self, sid, address, size):
        data, result, error = self.packet.readTxRx(sid, address, size)
        self.check(result, error, f"ID {sid} 读地址 {address}")
        if len(data) != size:
            raise RuntimeError(f"ID {sid} 反馈长度错误")
        return bytes(data)

    def read_int(self, sid, address, size):
        return int.from_bytes(self.read(sid, address, size), "little")

    def write_verified(self, sid, address, size, value):
        wire_value = position_word(value) if address == GOAL_POSITION and size == 2 else value
        payload = list(wire_value.to_bytes(size, "little"))
        result, error = self.packet.writeTxRx(sid, address, size, payload)
        self.check(result, error, f"ID {sid} 写地址 {address}")
        if self.read_int(sid, address, size) != wire_value:
            raise RuntimeError(f"ID {sid} 地址 {address} 回读与设定值不符")

    def status(self, sid):
        data = self.read(sid, PRESENT_POSITION, 15)
        return {
            "id": sid,
            "position_tick": signed_word(data[0:2]),
            "speed_rpm": round(signed_word(data[2:4]) * 0.732, 3),
            "voltage_v": data[6] / 10,
            "temperature_c": data[7],
            "status": data[9],
            "motor_current_ma": signed_word(data[13:15]) * 6.5,
            "moving": data[10],
            "active_target_tick": signed_word(data[11:13]),
        }

    def motion_status(self, sid):
        """Read actual enable/command settings as well as encoder feedback."""
        control = self.read(sid, TORQUE_ENABLE, 10)
        return {**self.status(sid), "torque_enabled": control[0],
                "goal_tick": signed_word(control[2:4]),
                "current_limit_ma": int.from_bytes(control[4:6], "little") * 6.5,
                "speed_limit_rpm": round(int.from_bytes(control[6:8], "little") * 0.732, 3),
                "torque_limit_percent": int.from_bytes(control[8:10], "little") / 10}

    def probe(self, sid):
        limits = self.read(sid, ANGLE_LIMITS, 4)
        addressing = self.read(sid, SERVO_ADDRESS, 4)
        firmware = self.read(sid, 0, 3)
        return {
            **self.status(sid),
            "model_number": self.read_int(sid, MODEL, 2),
            "mode": self.read_int(sid, MODE, 1),
            "resolution": self.read_int(sid, RESOLUTION, 1),
            "min_tick": int.from_bytes(limits[:2], "little"),
            "max_tick": int.from_bytes(limits[2:], "little"),
            "torque_enabled": self.read_int(sid, TORQUE_ENABLE, 1),
            "primary_id": addressing[0],
            "baud_code": addressing[1],
            "baud_rate": HLS_BAUD_RATES[addressing[1]] if addressing[1] < len(HLS_BAUD_RATES) else None,
            "secondary_id": addressing[2],
            "reply_level": addressing[3],
            "firmware_major": firmware[0], "firmware_minor": firmware[1],
            "endian": firmware[2],
            "full_angle_feedback": bool(self.read_int(sid, 18, 1) & 16),
        }

    def prepare(self, motors, model_number, *, stop_event=None):
        positions = {}
        for sid, motor in motors.items():
            if stop_event is not None and stop_event.is_set():
                return
            info = self.probe(sid)
            self.check_health(info)
            if info["model_number"] != model_number:
                raise RuntimeError(f"ID {sid} 型号编号与配置不符")
            if info["mode"] != 0 or info["resolution"] != 1:
                raise RuntimeError(f"ID {sid} 需要位置模式 0 和单圈分辨率系数 1")
            hardware_low, hardware_high = hardware_limits(info, motor.allow_multiturn)
            if info["primary_id"] != sid or info["reply_level"] != 1:
                raise RuntimeError(f"ID {sid} 主 ID 或应答级别错误")
            alias = info["secondary_id"]
            if alias != sid and alias in motors:
                raise RuntimeError(f"ID {sid} 的副 ID {alias} 与其他舵机冲突")
            low = max(motor.min_tick, hardware_low)
            high = min(motor.max_tick, hardware_high)
            position = info["position_tick"]
            if type(position) is int and not low <= position <= high:
                boundary = "低于下限" if position < low else "高于上限"
                difference = low - position if position < low else position - high
                raise ValueError(
                    f"ID {sid} 初始位置超出允许范围，本次启动已取消。\n"
                    f"实际反馈：{position} 编码；允许范围：{low}–{high} 编码。\n"
                    f"{boundary} {difference} 个编码（约 {difference / TICKS_PER_DEGREE:.2f}° 舵机轴角度）。\n"
                    "请先恢复到已标定的机构姿态；若零位已改变，确认机构处于定义的 0°姿态后，"
                    "重新「采集绑定零位」并保存配置。空载测试可使用当前静止位置作为测试零位。"
                )
            integer(position, low, high, f"ID {sid} 初始位置")
            if info["torque_enabled"]:
                raise RuntimeError(f"ID {sid} 已处于使能状态，请卸载后重新开始")
            self.limits[sid] = low, high
            positions[sid] = position

        for sid, motor in motors.items():
            if stop_event is not None and stop_event.is_set():
                return
            self.write_verified(sid, ACCELERATION, 1, 5)
            self.write_verified(sid, CURRENT_LIMIT, 2, math.floor(motor.current_ma / 6.5))
            self.write_verified(sid, SPEED_LIMIT, 2, round(motor.speed_rpm / 0.732))
            self.write_verified(sid, TORQUE_LIMIT, 2, 300)
        return positions

    def move(self, targets, *, stop_event=None):
        if targets.keys() != self.limits.keys():
            raise ValueError("目标舵机与初始化配置不符")
        for sid, tick in targets.items():
            integer(tick, *self.limits[sid], f"ID {sid} 目标位置")
        first_move = not self.enabled_ids
        if first_move:
            # Enabling can reset a firmware's current position command. Seed a
            # safe hold, confirm enable, then send the final targets to all IDs.
            # Do not enable against an old goal left by an earlier operation.
            holds = {}
            for sid in targets:
                if stop_event is not None and stop_event.is_set():
                    return
                reading = self.status(sid)
                self.check_health(reading)
                holds[sid] = integer(reading["position_tick"], *self.limits[sid], f"ID {sid} 保持位置")
                if abs(reading["speed_rpm"]) > 1 or self.read_int(sid, TORQUE_ENABLE, 1) != 0:
                    raise RuntimeError(f"ID {sid} 初始化前需要静止且扭矩关闭")
            for sid, position in holds.items():
                if stop_event is not None and stop_event.is_set():
                    return
                self.enabled_ids.add(sid)
                self.write_verified(sid, GOAL_POSITION, 2, position)
                if stop_event is not None and stop_event.is_set():
                    return
                torque = self.read_int(sid, TORQUE_ENABLE, 1)
                if torque == 0:
                    self.write_verified(sid, TORQUE_ENABLE, 1, 1)
                elif torque != 1:
                    raise RuntimeError(f"ID {sid} 初始化时扭矩状态异常：{torque}")
        if len(targets) == 1:
            # A lone servo needs no broadcast: require its write response and
            # register readback, as in the successful single-servo bench test.
            if stop_event is not None and stop_event.is_set():
                return
            self.enabled_ids.update(targets)
            for sid, tick in targets.items():
                self.write_verified(sid, GOAL_POSITION, 2, tick)
        else:
            try:
                for sid, tick in targets.items():
                    if not self.sync.addParam(sid, list(position_word(tick).to_bytes(2, "little"))):
                        raise RuntimeError(f"无法把 ID {sid} 加入同步写")
                # Track all possibly enabled IDs before the first outgoing byte.
                if stop_event is not None and stop_event.is_set():
                    return
                self.enabled_ids.update(targets)
                self.check(self.sync.txPacket(), 0, "同步写目标位置")
            finally:
                self.sync.clearParam()
        if first_move:
            for sid, tick in targets.items():
                if stop_event is not None and stop_event.is_set():
                    return
                if self.read_int(sid, GOAL_POSITION, 2) != position_word(tick):
                    raise RuntimeError(f"ID {sid} 未确认收到初始位置")
                if self.read_int(sid, TORQUE_ENABLE, 1) != 1:
                    raise RuntimeError(f"ID {sid} 目标发送后扭矩已关闭；本次停止，不自动重新使能")

    @staticmethod
    def check_health(info):
        if info["status"] or not 4.5 <= info["voltage_v"] <= 7.4 or info["temperature_c"] >= 65:
            reasons = []
            voltage, temperature = info["voltage_v"], info["temperature_c"]
            if voltage < 4.5:
                reasons.append(f"低电压：反馈 {voltage:.1f} V，低于程序保护下限 4.5 V")
            elif voltage > 7.4:
                reasons.append(f"电压过高：反馈 {voltage:.1f} V，超过程序保护上限 7.4 V")
            elif not 4.5 <= voltage <= 7.4:
                reasons.append("电压反馈不是有效数值")
            if temperature >= 65:
                reasons.append(f"温度过高：反馈 {temperature:g}°C，达到程序保护上限 65°C")
            if info["status"]:
                reasons.append(f"舵机状态字异常：{info['status']}")
            hint = "请核对外部电源输出、限流设置，以及 Hub 和供电线在转动时的电压。" if not 4.5 <= voltage <= 7.4 else "请排查负载、温升及设备反馈后重试。"
            raise RuntimeError(f"ID {info.get('id', '?')} 舵机健康保护：\n"
                               + "\n".join(reasons) + "\n本次操作中止，退出时关闭本程序已使能的舵机。\n"
                               + hint + f"\n\n原始反馈：{info}")

    def close(self):
        failed = []
        try:
            for sid in sorted(self.enabled_ids):
                try:
                    # 串口异常中断时，SDK 可能仍保留 busy 标记。
                    self.port.is_using = False
                    self.write_verified(sid, TORQUE_ENABLE, 1, 0)
                except Exception as exc:
                    failed.append(f"ID {sid}: {exc}")
        finally:
            self.port.closePort()
        if failed:
            raise RuntimeError("未确认扭矩关闭，请切断舵机电源。 " + ", ".join(failed))


def signed_word(data):
    value = int.from_bytes(data, "little")
    return -(value & 0x7FFF) if value & 0x8000 else value


def position_word(tick):
    """Encode HLS signed magnitude, never modulo a turn or use two's complement."""
    integer(tick, MULTITURN_MIN_TICK, MULTITURN_MAX_TICK, "HLS 位置编码")
    return abs(tick) | (0x8000 if tick < 0 else 0)


def hardware_limits(info, allow_multiturn=False):
    """Keep calibrated software bounds even when HLS hardware limits are 0/0."""
    low, high = info["min_tick"], info["max_tick"]
    if low == high == 0 and allow_multiturn:
        if (info.get("firmware_major") == 3 and 43 <= info.get("firmware_minor", 0) <= 59
                and info.get("endian") == 0 and info.get("full_angle_feedback") is True):
            return MULTITURN_MIN_TICK, MULTITURN_MAX_TICK
        raise RuntimeError(f"ID {info['id']} 多圈模式需要已验证的 HLS 固件与全角度反馈")
    if not 0 <= low < high <= 4095:
        raise RuntimeError(f"ID {info['id']} 需要有效的单圈角度范围或已采集的多圈绑定零位")
    return low, high


def calibrate_bindings(config, readings):
    """Create a complete new binding configuration from read-only measurements."""
    if not config.motors or readings.keys() != config.motors.keys():
        raise ValueError("请完整读取已绑定的舵机；未绑定的舵机无需连接")
    motors, model = [], None
    for sid, motor in config.motors.items():
        info = readings[sid]
        HlsBus.check_health(info)
        if (info["id"] != sid or info["primary_id"] != sid or info["reply_level"] != 1
                or info["torque_enabled"] or info["mode"] != 0 or info["resolution"] != 1
                or abs(info["speed_rpm"]) > 1):
            raise ValueError(f"ID {sid} 需要主 ID 正确、应答级别 1、静止卸载、位置模式 0 和分辨率 1")
        if info["secondary_id"] != sid and info["secondary_id"] in config.motors:
            raise ValueError(f"ID {sid} 的副 ID 与其他已绑定舵机冲突")
        reported_model = integer(info["model_number"], 1, 65535, "model_number")
        if model is not None and model != reported_model:
            raise ValueError("已绑定舵机的型号编号不一致")
        model = reported_model
        multiturn = info["min_tick"] == info["max_tick"] == 0
        hardware_low, hardware_high = hardware_limits(info, multiturn)
        tick = integer(info["position_tick"], hardware_low, hardware_high, f"ID {sid} 零位")
        offset = tick - motor.zero_tick
        low, high = motor.min_tick + offset, motor.max_tick + offset
        integer(low, hardware_low, tick, f"ID {sid} 移动后的软下限")
        integer(high, tick, hardware_high, f"ID {sid} 移动后的软上限")
        if low >= high:
            raise ValueError(f"ID {sid} 请先填写非零的软限位范围")
        motors.append(asdict(replace(motor, zero_tick=tick, min_tick=low, max_tick=high,
                                     allow_multiturn=multiturn)))
    data = config.to_data()
    data.update(calibrated=True, allow_unbound_joints=True, model_number=model, motors=motors)
    return HandConfig.from_data(data)


def pose_timeout_seconds(motors, positions, targets):
    """Allow for the acceleration/deceleration set by HlsBus.prepare, plus settling."""
    longest = 0.0
    acceleration = 5 * 8.79  # SRAM acceleration 5, degrees/s²
    for sid, tick in targets.items():
        distance = abs(tick - positions[sid]) / TICKS_PER_DEGREE
        speed = round(motors[sid].speed_rpm / 0.732) * 0.732 * 6
        if distance < speed * speed / acceleration:
            duration = 2 * math.sqrt(distance / acceleration)
        else:
            duration = distance / speed + speed / acceleration
        longest = max(longest, duration)
    return max(3.0, longest * 1.5 + 2.0)


def pose_timeout_message(positions, targets, readings, timeout_s, *, reason="未在预计时间内到位"):
    lines = [f"绑定舵机{reason}（本轮等待上限 {timeout_s:.1f} 秒）；本次已结束并进入关闭扭矩流程。"]
    for sid, target in list(targets.items())[:5]:
        info = readings.get(sid)
        if info is None:
            lines.append(f"ID {sid}：目标 {target}，未取得本轮位置反馈。")
            continue
        position = info["position_tick"]
        movement = (position - positions[sid]) / TICKS_PER_DEGREE
        error = (target - position) / TICKS_PER_DEGREE
        state = "未观察到有效转动" if abs(movement) <= .5 and abs(error) > .5 else "已转动但未稳定到位"
        lines.append(f"ID {sid}：{state}；实际位移 {movement:+.2f}°，剩余偏差 {error:+.2f}°。")
        torque = info.get("torque_enabled")
        if torque is not None:
            lines.append(f"运行中扭矩：{'开启' if torque == 1 else '关闭' if torque == 0 else torque}；"
                         f"目标编码 {target}，指令回读 {info.get('goal_tick', '—')}，反馈 {position}。")
        if "voltage_v" in info:
            lines.append(f"电压 {info['voltage_v']:.1f} V，相电流 {info.get('motor_current_ma', '—')} mA，"
                         f"速度 {info.get('speed_rpm', '—')} RPM。")
    if len(targets) > 5:
        lines.append("其余舵机的完整反馈见运行日志。")
    return "\n".join(lines)


@contextmanager
def serial_bus(port, baud):
    bus = HlsBus(port, baud)
    try:
        yield bus
    finally:
        bus.close()


def receive_frame(connection, buffer, timeout=FRAME_TIMEOUT, *, stop_event=None):
    deadline = time.monotonic() + timeout
    while b"\n" not in buffer:
        if stop_event is not None and stop_event.is_set():
            return None, b""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("接收角度超时")
        connection.settimeout(min(0.1, remaining) if stop_event is not None else remaining)
        try:
            chunk = connection.recv(MAX_FRAME_BYTES - len(buffer))
        except socket.timeout:
            if stop_event is None:
                raise
            continue
        if not chunk:
            if buffer:
                raise ValueError("收到不完整的 JSON 帧")
            return None, b""
        buffer += chunk
        if len(buffer) >= MAX_FRAME_BYTES and b"\n" not in buffer:
            raise ValueError("JSON 帧过长")
    line, _, buffer = buffer.partition(b"\n")
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("JSON 帧须为对象")
    return message, buffer


def send_frame(connection, message):
    connection.settimeout(FRAME_TIMEOUT)
    connection.sendall(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")


class CommandSession:
    """实机关节帧的映射、发送及反馈处理。"""

    def __init__(self, config, bus, stop_event=None, *, nearby_start=False):
        self.config = config
        self.bus = bus
        self.stop_event = stop_event
        self.nearby_start = nearby_start
        self.frame = 0
        self.next_command = 0
        self.next_status = 0
        self.readings = []
        self.initial_positions = None

    def stopped(self):
        return self.stop_event is not None and self.stop_event.is_set()

    def get_hand_data(self, *, check=True):
        readings = []
        for sid in self.config.motors:
            if self.stopped():
                break
            info = self.bus.status(sid)
            if check and isinstance(self.bus, HlsBus):
                self.bus.check_health(info)
                integer(info["position_tick"], *self.bus.limits[sid], f"ID {sid} 反馈位置")
            readings.append(info)
        return readings

    def prepare(self):
        if self.initial_positions is None:
            positions = self.bus.prepare(self.config.motors, self.config.model_number,
                                         stop_event=self.stop_event)
            if self.stopped():
                return False
            self.initial_positions = positions
        return True

    def set_joint_positions(self, angles):
        start = time.monotonic()
        targets = self.config.targets(angles)
        if not targets:
            raise ValueError("当前没有已绑定的舵机，不发送姿态")
        if self.stopped():
            return None
        if not self.prepare():
            return None
        if self.frame == 0 and self.nearby_start:
            for sid, tick in targets.items():
                if abs(tick - self.initial_positions[sid]) > round(5 * TICKS_PER_DEGREE):
                    raise ValueError(f"ID {sid} 首帧与当前位置差超过约 5°，请先对齐初始姿态")
        delay = self.next_command - time.monotonic()
        if delay > 0:
            if self.stop_event is None:
                time.sleep(delay)
            elif self.stop_event.wait(delay):
                return None
        self.next_command = time.monotonic() + 0.05
        self.bus.move(targets, stop_event=self.stop_event)
        fresh = time.monotonic() >= self.next_status
        if fresh:
            self.readings = self.get_hand_data()
            self.next_status = time.monotonic() + 0.15
        if self.stopped():
            return None
        self.frame += 1
        motor_angles = {sid: detail["motor_deg"] for sid, detail in self.config.describe_mapping(angles).items()}
        return {"ok": True, "frame": self.frame, "target_tick": targets, "motor_target_deg": motor_angles, "servo_status": self.readings,
                "feedback_fresh": fresh, "process_ms": round((time.monotonic() - start) * 1000, 1)}


def serve(config, bus, tcp_port, *, stop_event=None, on_frame=None, notice=None,
          nearby_start=False):
    notice = notice or (lambda message: print(message, flush=True))
    session = CommandSession(config, bus, stop_event, nearby_start=nearby_start)
    with socket.socket() as listener:
        listener.bind((HOST, tcp_port))
        listener.listen(1)
        notice(f"监听 {HOST}:{tcp_port} · 等待视觉程序连接")
        if stop_event is not None:
            listener.settimeout(0.1)
        while not session.stopped():
            try:
                connection, _ = listener.accept()
                break
            except socket.timeout:
                if stop_event is None:
                    raise
        else:
            return
        with connection:
            buffer = b""
            notice("视觉程序已连接 · 等待有效关节角度")
            while not session.stopped():
                try:
                    message, buffer = receive_frame(connection, buffer, stop_event=stop_event)
                    if message is None:
                        return
                    if set(message) != {"joints_deg"}:
                        raise ValueError("消息须为包含 joints_deg 的 JSON 对象")
                    angles = message["joints_deg"]
                    reply = session.set_joint_positions(angles)
                    if reply is None:
                        return
                    if on_frame is not None:
                        on_frame(angles, reply)
                    send_frame(connection, reply)
                except (ValueError, TypeError, RuntimeError) as exc:
                    try:
                        send_frame(connection, {"ok": False, "error": str(exc)})
                    except OSError:
                        pass
                    raise


def jog(bus, sid, degrees, *, report=None, stop_event=None):
    info = bus.probe(sid)
    position = info["position_tick"]
    multiturn = info["min_tick"] == info["max_tick"] == 0
    hardware_low, hardware_high = hardware_limits(info, multiturn)
    low = max(hardware_low, position - round(5 * TICKS_PER_DEGREE))
    high = min(hardware_high, position + round(5 * TICKS_PER_DEGREE))
    motor = Motor(sid, position, low, high, {}, 250, 5, allow_multiturn=multiturn)
    bus.prepare({sid: motor}, info["model_number"], stop_event=stop_event)
    if stop_event is not None and stop_event.is_set():
        return
    bus.move({sid: position + round(degrees * TICKS_PER_DEGREE)}, stop_event=stop_event)
    for _ in range(5):
        if stop_event is not None and stop_event.is_set():
            break
        reading = bus.status(sid)
        bus.check_health(reading)
        if report is None:
            print(json.dumps(reading, ensure_ascii=False))
        else:
            report(reading)
        if stop_event is None:
            time.sleep(0.15)
        elif stop_event.wait(0.15):
            break


def main():
    parser = argparse.ArgumentParser(description="飞特 HLS 实机控制")
    parser.add_argument("command", choices=("serve", "jog"))
    parser.add_argument("--config", type=Path, default=ROOT / "index.json")
    parser.add_argument("--port", help="FE-URT-2 串口，例如 COM4")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--degrees", type=float, default=3)
    parser.add_argument("--enable-motion", action="store_true")
    args = parser.parse_args()
    if not args.port:
        parser.error("请指定真实串口 --port COM号")
    if not args.enable_motion:
        parser.error("实机运动须显式加 --enable-motion")
    if args.command == "jog":
        integer(args.id, 0, 253, "ID")
        number(args.degrees, -5, 5, "--degrees")
        with serial_bus(args.port, args.baud) as bus:
            jog(bus, args.id, args.degrees)
        return
    config = HandConfig(args.config)
    if not config.calibrated or not config.motors:
        parser.error("实机服务须加载已采集零位且含绑定舵机的配置")
    integer(config.model_number, 1, 65535, "model_number")
    with serial_bus(args.port, args.baud) as bus:
        serve(config, bus, args.tcp_port, nearby_start=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        sys.exit(f"停止：{exc}")

