#!/usr/bin/env python3
"""第一次接触真实硬件：一层一层验证，每层给结论。

这个脚本回答一个问题：**"我的代码到底能不能控制真舵机？"**
它不猜，它逐层发真命令、看真回应，第一个失败的层就是当前卡住的地方。

用法：
    python tools/first_contact.py --list                  # 只列出可用串口
    python tools/first_contact.py --port COM3             # 只读层：开串口 + PING
    python tools/first_contact.py --port COM3 --move      # 加上单轴小角度往返 + 12 路整帧

层次：
    L1 打开串口       端口存在、没被别的程序占用
    L2 PING / PONG    固件在跑、波特率对、双向通信通
    L3 单轴小角度     固件真的驱了舵机（唯一需要 --move 的层）
    L4 12 路整帧      完整协议帧能被解析，ACK 计数正确

安全前提（--move 之前必须逐条确认）：
    * 舵机电源来自外部限流电源，**绝不接 Uno 的 5V 引脚**
    * 舵机已固定，舵机臂上**没有装任何机构**
    * 舵机电源地与 Uno 地已正确共地
    * 手边有一个能直接切断舵机电源的物理开关
    * 只用一个小角度（默认 10 度）。机械限位、供电、防护和急停未验证前，
      不要在机构上跑，也不要用 hand_tester --full-range
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.motor_control.serial_controller import (  # noqa: E402
    RealSerial,
    SerialController,
    is_ping_response,
    parse_ack,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "hand_config.yaml"

# A deliberate guard: at first contact nothing is calibrated, so only a small
# excursion from the rest pose is allowed.  Anything larger needs a real
# mechanical limit and a bench supply, which this script cannot verify.
MAX_DELTA_WITHOUT_MECHANISM = 30

# Time allowed for a small move to physically complete before the next command.
SETTLE_SECONDS = 0.8


def _joint_names(config: dict) -> List[str]:
    pins = (config.get("servos") or {}).get("pins") or {}
    return [str(name) for name in pins]


def _default_angle(config: dict) -> int:
    limits = (config.get("servos") or {}).get("angle_limits") or {}
    try:
        return int(limits.get("default_angle", 90))
    except (TypeError, ValueError):
        return 90


def list_ports() -> int:
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("[FAIL] pyserial 未安装。运行: pip install pyserial")
        return 1

    ports = sorted(lp.comports(), key=lambda p: p.device)
    if not ports:
        print("没有发现任何串口。")
        print("  - Uno 插上了吗？USB 线是数据线还是只有电的充电线？")
        print("  - Windows 设备管理器 -> 端口(COM 和 LPT) 里有没有新增项？")
        print("  - 装了 CH340 驱动吗？（国产 Uno 大多是 CH340 芯片）")
        return 1

    print(f"发现 {len(ports)} 个串口：")
    for port in ports:
        print(f"  {port.device:<8} {port.description}")
    print("\nUno 通常是描述里带 'CH340' 或 'Arduino' 的那个。")
    return 0


class Layers:
    """Runs the layers in order and remembers where it stopped."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str) -> bool:
        self.results.append((name, ok, detail))
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    def first_failure(self) -> Optional[Tuple[str, str]]:
        for name, ok, detail in self.results:
            if not ok:
                return name, detail
        return None

    def report(self) -> None:
        print("\n" + "=" * 62)
        print("结论")
        print("=" * 62)
        for name, ok, detail in self.results:
            print(f"  {'通过' if ok else '未通过'}  {name}")

        failure = self.first_failure()
        if failure is None:
            print("\n全部通过。下一步：")
            print("  1. python -m src.diagnostics.hand_tester --real list")
            print("  2. 装第一个手指件之前，先在台架上量 stall 电流与电压跌落")
            print("  3. python -m src.main --real     # 摄像头 -> 映射 -> 舵机（S2）")
            return

        name, detail = failure
        print(f"\n第一个卡住的层：{name}（{detail}）")
        print(_DIAGNOSIS.get(name, "  （无预置诊断，请把上面的原始输出贴出来）"))


_DIAGNOSIS = {
    "L1 打开串口": (
        "  端口打不开。依次排查：\n"
        "  - 端口号对不对（--list 看）\n"
        "  - Arduino IDE 的串口监视器还开着吗？同一端口只能被一个程序占用\n"
        "  - 拔掉重插，确认设备管理器里端口出现又消失"
    ),
    "L2 PING / PONG": (
        "  串口能开但固件没应答。依次排查：\n"
        "  - 固件真的烧进去了吗？（板载 LED 有没有按预期闪）\n"
        "  - 波特率：清单里是 115200，固件里 Serial.begin() 是同一个数吗\n"
        "  - Uno 每次打开串口会复位，脚本已等 2 秒；若仍无应答，手动重试一次\n"
        "  - 如果收到的是乱码，基本可以确定是波特率不匹配"
    ),
    "L3 单轴小角度": (
        "  PING 通了但舵机不动。这是**接线或供电**问题，不是代码问题：\n"
        "  - 舵机信号线（橙/黄）接的是清单里那一根引脚吗\n"
        "  - 舵机电源线（红）接的是外部电源，**不是 Uno 的 5V**\n"
        "  - 外部电源和 Uno 共地了吗？（这是最常见的漏项）\n"
        "  - 电源打开了吗？万用表量一下舵机电源端是不是 5V"
    ),
    "L4 12 路整帧": (
        "  单轴能动但整帧不过。多半是固件的帧解析或 ACK 计数问题：\n"
        "  - 看 ACK 里的数字是不是 12（不是 12 说明固件丢字段了）\n"
        "  - 固件里的 SERVO_NAMES 和 config 的 servos.pins 键名是否完全一致\n"
        "  - 先跑 python -m src.diagnostics.hardware_validator 交叉比对"
    ),
}


def layer_open(port: str, baudrate: int) -> Tuple[Optional[SerialController], str]:
    interface = RealSerial(port=port, baudrate=baudrate)
    controller = SerialController(interface=interface)
    if controller.connect():
        return controller, f"{port} @ {baudrate}"
    return None, f"打不开 {port}"


def layer_ping(controller: SerialController) -> Tuple[bool, str]:
    response = controller.send_and_receive("PING", timeout=1.5)
    if is_ping_response(response):
        return True, "收到 PONG"
    return False, f"期望 PONG，收到 {response!r}"


def layer_single_axis(
    controller: SerialController, joint: str, base: int, delta: int
) -> Tuple[bool, str]:
    outbound = min(180, base + delta)
    for label, target in (("移出", outbound), ("回位", base)):
        response = controller.send_and_receive(f"{joint}:{target}", timeout=1.5, seq=1)
        ack = parse_ack(response)
        if ack is None or ack[1] != 1:
            return False, f"{label} {joint}->{target} 未收到有效 ACK（收到 {response!r}）"
        time.sleep(SETTLE_SECONDS)
    return True, f"{joint} {base} -> {outbound} -> {base} 往返成功"


def layer_full_frame(
    controller: SerialController, joints: List[str], base: int
) -> Tuple[bool, str]:
    frame = ",".join(f"{joint}:{base}" for joint in joints)
    response = controller.send_and_receive(frame, timeout=2.0, seq=2)
    ack = parse_ack(response)
    if ack is None:
        return False, f"未收到有效 ACK（收到 {response!r}）"
    if ack[1] != len(joints):
        return False, f"ACK 计数 {ack[1]} != 发出的 {len(joints)} 路"
    return True, f"{len(joints)} 路整帧被接收，ACK 计数正确"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="第一次接触真实硬件：逐层验证，给结论",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="只列出可用串口后退出")
    parser.add_argument("--port", "-p", type=str, default=None, help="串口名，例如 COM3")
    parser.add_argument("--config", "-c", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--baudrate", type=int, default=None, help="默认取配置里的值")
    parser.add_argument(
        "--move",
        action="store_true",
        help="允许 L3/L4 真的驱动舵机。不加则只跑只读层",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=10,
        help=f"单轴测试的偏移角度（默认 10，上限 {MAX_DELTA_WITHOUT_MECHANISM}）",
    )
    args = parser.parse_args()

    if args.list:
        return list_ports()

    try:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"[FAIL] 读不了配置 {args.config}: {exc}")
        return 1

    serial_cfg = config.get("serial") or {}
    port = args.port or serial_cfg.get("port")
    baudrate = args.baudrate or int(serial_cfg.get("baudrate", 115200))
    joints = _joint_names(config)
    base = _default_angle(config)

    if not port:
        print("[FAIL] 没有端口。用 --list 看真实端口，或用 --port 指定。")
        return 1
    if not joints:
        print("[FAIL] 配置里 servos.pins 是空的，无法确定轴名。")
        return 1

    print("=" * 62)
    print(f"第一次接触  port={port}  baud={baudrate}  轴数={len(joints)}  基准角={base}")
    print("=" * 62)

    layers = Layers()

    controller, detail = layer_open(port, baudrate)
    if not layers.record("L1 打开串口", controller is not None, detail):
        layers.report()
        return 1
    assert controller is not None

    try:
        ok, detail = layer_ping(controller)
        if not layers.record("L2 PING / PONG", ok, detail):
            layers.report()
            return 1

        if not args.move:
            print("\n（只读层到此为止。确认电源与机械安全后加 --move 继续 L3/L4。）")
            layers.report()
            return 0

        delta = max(1, min(abs(args.delta), MAX_DELTA_WITHOUT_MECHANISM))
        if args.delta != delta:
            print(f"  偏移角被限制为 {delta} 度（上限 {MAX_DELTA_WITHOUT_MECHANISM}）")
        print("\n  即将驱动舵机。确认：外部限流电源 / 已共地 / 舵机臂上无机构 / 有物理开关")

        joint = joints[0]
        ok, detail = layer_single_axis(controller, joint, base, delta)
        if not layers.record("L3 单轴小角度", ok, detail):
            layers.report()
            return 1

        ok, detail = layer_full_frame(controller, joints, base)
        layers.record("L4 12 路整帧", ok, detail)

    finally:
        controller.disconnect()

    layers.report()
    return 0 if layers.first_failure() is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
