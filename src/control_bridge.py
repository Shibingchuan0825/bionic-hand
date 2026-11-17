"""连接现有视觉角度与 HLS 控制端，不改变原来的识别/关节计算模块。"""
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

CONTROL_DIR = Path(__file__).resolve().parents[1] / "上位机"
if not (CONTROL_DIR / "hand_integration.py").is_file():
    raise ImportError("请保留仓库根目录下的「上位机」目录，统一程序需要其中的控制模块")
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

from hand_control import initial_angles
from .hand_mapper.joint_mapper import JointMapper


FINGER_BASES = {"index": 5, "middle": 9, "ring": 13, "little": 17}
SOURCE_JOINTS = {
    **{f"{finger}_{axis}_flex": f"{source}_{axis}"
       for finger, source in (("index", "index"), ("middle", "middle"),
                              ("ring", "ring"), ("little", "pinky"))
       for axis in ("mcp", "pip")},
    "thumb_mcp_flex": "thumb_mcp", "thumb_ip_flex": "thumb_ip",
}
SUPPORTED_JOINTS = frozenset(SOURCE_JOINTS) | {f"{finger}_dip_flex" for finger in FINGER_BASES}


def extract_joint_features(angles, landmarks):
    """沿用原 2D 角度；DIP 从同一批关键点直接计算，不用 PIP 代替。"""
    if angles is None or landmarks is None or not landmarks.is_valid():
        return None
    source = angles.to_dict()
    features = {joint: float(source[name]) for joint, name in SOURCE_JOINTS.items()}
    points = landmarks.landmarks_px
    for finger, base in FINGER_BASES.items():
        before, center, after = points[base + 1:base + 4]
        if (np.linalg.norm(before[:2] - center[:2]) < 1e-6
                or np.linalg.norm(after[:2] - center[:2]) < 1e-6):
            raise ValueError(f"{finger} DIP 关键点重合")
        features[f"{finger}_dip_flex"] = JointMapper._angle_between(before, center, after)
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("视觉角度含无效数值")
    return features


class VisionControlBridge:
    """识别线程直接发布到控制输入；Tk 只读状态，不负责逐帧转发。"""

    def __init__(self):
        self._lock = threading.RLock()
        self.config = self.source = None
        self.neutral = None
        self.neutral_hand = None
        self.latest = None
        self.gain = 1.0
        self.message = "等待识别参数"
        self.targets = None
        self.limited = set()
        self.last_valid_at = None

    def configure(self, config, source):
        with self._lock:
            if self.source is not None:
                self.source.invalidate("绑定配置已变更")
            self.config, self.source = config, source
            self.neutral = self.neutral_hand = self.latest = self.targets = None
            self.limited = set()
            self.last_valid_at = None
            self.message = "绑定已加载；请在镜头前采集视觉零姿态"

    def unsupported_bindings(self):
        with self._lock:
            bound = {joint for motor in self.config.motors.values() for joint in motor.weights}
            return sorted(bound - SUPPORTED_JOINTS)

    def set_gain(self, gain):
        if type(gain) not in (float, int) or not math.isfinite(gain) or not .05 <= gain <= 2:
            raise ValueError("视觉角度比例须在 0.05 到 2 之间")
        with self._lock:
            self.gain = gain
            self.neutral = self.targets = None
            self.source.invalidate("视觉角度比例已变更，请重新采集视觉零姿态")
            self.message = "比例已更新；请重新采集视觉零姿态"

    def capture_neutral(self):
        with self._lock:
            if self.latest is None or time.monotonic() - self.latest[2] > self.source.max_age:
                raise ValueError("请先把手放到镜头前，等待有效识别")
            self.neutral, self.neutral_hand = self.latest[0].copy(), self.latest[1]
            self.source.invalidate("视觉零姿态已重新采集")
            self.targets = None
            self.message = "视觉零姿态已采集；等待下一帧自动传入控制端"

    def tracking_lost(self, reason="未检测到有效人手"):
        with self._lock:
            self.latest = self.targets = None
            self.last_valid_at = None
            self.limited = set()
            self.message = reason
            if self.source is not None:
                self.source.invalidate(reason)

    def publish(self, features, handedness, captured_at):
        with self._lock:
            if self.source is None:
                return
            if features is None:
                # Brief missed detections must not replay old poses. Keep the
                # input idle for at most 250 ms, then latch a stop for this run.
                self.latest = self.targets = None
                if self.last_valid_at is None or time.monotonic() - self.last_valid_at >= .25:
                    self.tracking_lost()
                else:
                    self.message = "暂未检测到手，暂停更新关节参数"
                return
            features = features.copy()
            if (not SUPPORTED_JOINTS <= features.keys()
                    or not all(type(value) in (int, float) and math.isfinite(value)
                               for value in features.values())):
                self.tracking_lost("视觉参数不完整或无效")
                return
            if time.monotonic() - captured_at > self.source.max_age:
                self.tracking_lost("视觉处理延迟过大，未发送旧帧")
                return
            self.latest = features, handedness, captured_at
            self.last_valid_at = captured_at
            if self.neutral is None:
                self.message = "已识别到手；请采集视觉零姿态"
                return
            if handedness != self.neutral_hand:
                self.neutral = self.neutral_hand = None
                self.tracking_lost("识别手已切换，请重新采集视觉零姿态")
                return
            targets = initial_angles(self.config)
            limited = set()
            for joint in self.config.joints.keys() & SUPPORTED_JOINTS:
                # Old vision uses interior angles (straight ~= 180). The
                # captured open pose defines robot zero, so flexion is positive.
                value = (self.neutral[joint] - features[joint]) * self.gain
                low, high = self.config.joints[joint]
                target = max(low, min(high, value))
                if target != value:
                    limited.add(joint)
                targets[joint] = round(target, 3)
            missing = self.unsupported_bindings()
            self.targets, self.limited = targets, limited
            if missing:
                self.source.invalidate("绑定包含暂无视觉参数的关节：" + ", ".join(missing))
                self.message = "部分绑定暂无视觉参数，请调整绑定后跟随"
                return
            try:
                self.source.submit_joints(targets, captured_at=captured_at)
            except (ValueError, TypeError) as exc:
                self.message = "控制端拒绝目标：" + str(exc)
            else:
                self.message = "视觉角度已自动传入控制端" + (" · 部分关节已限幅" if limited else "")

    def status(self):
        with self._lock:
            return {
                "neutral_ready": self.neutral is not None,
                "message": self.message,
                "features": self.latest[0].copy() if self.latest is not None else None,
                "captured_at": self.latest[2] if self.latest is not None else None,
                "hand": self.latest[1] if self.latest is not None else "—",
                "targets": self.targets.copy() if self.targets is not None else None,
                "limited": self.limited.copy(),
                "unsupported": self.unsupported_bindings() if self.config is not None else [],
            }
