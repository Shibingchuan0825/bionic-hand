"""A captured return pose and a bounded return on an already owned HLS bus."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

from hand_control import (HlsBus, SPEED_LIMIT, integer, pose_timeout_message,
                          pose_timeout_seconds)


def binding_signature(config):
    return hashlib.sha256(json.dumps(config.to_data(), sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class HomePose:
    def __init__(self, config, angles, *, visual_features=None):
        if not config.calibrated or not config.motors:
            raise ValueError("请先完成舵机关节绑定与零位标定")
        config.targets(angles)
        self.angles = angles.copy()
        self.signature = binding_signature(config)
        self.visual_features = dict(visual_features or {})

    def targets(self, config):
        if self.signature != binding_signature(config):
            raise ValueError("绑定、零位或限位已变更，请重新设置默认归位姿态")
        return config.targets(self.angles)

    def save(self, path):
        path = Path(path)
        data = dict(version=1, binding_signature=self.signature, joints_deg=self.angles,
                    visual_features=self.visual_features)
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, config, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if (not isinstance(data, dict) or data.get("version") != 1
                or data.get("binding_signature") != binding_signature(config)):
            raise ValueError("归位配置与当前绑定不匹配，请重新采集伸直姿态")
        return cls(config, data["joints_deg"], visual_features=data.get("visual_features"))


def return_to_home(config, bus, home, *, emergency, on_frame=None, notice=None):
    """Never open a port or enable an unowned motor; caller's context unloads."""
    targets = home.targets(config)
    owned = set(getattr(bus, "enabled_ids", ()))
    if emergency.is_set() or not owned:
        return False
    if owned != set(targets):
        raise RuntimeError("归位取消：本次使能的舵机不完整，直接进入卸载")
    notice = notice or (lambda message: None)
    positions = {}
    reader = getattr(bus, "motion_status", bus.status)

    def read_feedback():
        readings = {}
        for sid in targets:
            if emergency.is_set():
                return None
            info = reader(sid)
            HlsBus.check_health(info)
            if info.get("torque_enabled") != 1:
                raise RuntimeError(f"ID {sid} 归位取消：扭矩已关闭，不自动重新使能")
            integer(info["position_tick"], *bus.limits[sid], f"ID {sid} 归位反馈")
            readings[sid] = info
        return readings

    readings = read_feedback()
    if readings is None:
        return False
    positions = {sid: info["position_tick"] for sid, info in readings.items()}
    motors = {sid: replace(motor, speed_rpm=max(1, math.floor(min(motor.speed_rpm, 5) / .732)) * .732)
              for sid, motor in config.motors.items()}
    # Change only speed, downward. Keep the prepared current/torque limits.
    for sid, motor in motors.items():
        integer(targets[sid], *bus.limits[sid], f"ID {sid} 归位目标")
        if emergency.is_set():
            return False
        bus.write_verified(sid, SPEED_LIMIT, 2, round(motor.speed_rpm / .732))
    notice("正在自动归位到已保存的伸直姿态；Esc 可立即中止并卸载")
    bus.move(targets, stop_event=emergency)
    timeout = min(20.0, pose_timeout_seconds(motors, positions, targets))
    deadline, settled, frame = time.monotonic() + timeout, 0, 0
    while not emergency.is_set():
        if time.monotonic() >= deadline:
            raise RuntimeError(pose_timeout_message(positions, targets, readings, timeout, reason="归位超时"))
        readings = read_feedback()
        if readings is None:
            return False
        for sid, info in readings.items():
            if info.get("goal_tick") != targets[sid]:
                raise RuntimeError(f"ID {sid} 归位目标指令未保持，本次停止并卸载")
        settled = settled + 1 if all(abs(readings[sid]["position_tick"] - tick) <= 6
                                    and abs(readings[sid]["speed_rpm"]) <= 1
                                    for sid, tick in targets.items()) else 0
        frame += 1
        if on_frame is not None:
            on_frame(home.angles.copy(), dict(ok=True, frame=frame, target_tick=targets.copy(),
                servo_status=list(readings.values()), feedback_fresh=True, process_ms=0,
                feedback_monotonic_s=time.monotonic(), execution_phase="homing"))
        if settled >= 3:
            notice("默认姿态归位已到位，正在关闭扭矩")
            return True
        emergency.wait(.05)
    return False
