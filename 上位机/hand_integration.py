"""视觉程序的进程内控制接口；只接收机器人关节角度，不计算视觉角度。"""
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

from hand_control import CommandSession, HandConfig, integer, serial_bus
from hand_home import HomePose, return_to_home


class TrackingLostError(ValueError):
    """A visual-input interruption, distinct from a hardware/limit failure."""


class FrameExpiredError(TimeoutError):
    pass


@dataclass(frozen=True)
class JointFrame:
    sequence: int
    angles: dict
    captured_at: float


class JointInput:
    """只保存最新有效帧；无效帧和跟踪丢失会使当前联动失效。"""

    def __init__(self, config, *, max_age=0.5):
        if type(max_age) not in (int, float) or not math.isfinite(max_age) or not 0 < max_age <= 1:
            raise ValueError("max_age 必须大于 0 且不超过 1 秒")
        self.config = HandConfig.from_data(config.to_data())
        self.max_age = max_age
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = 0
        self._failure = 0
        self._reason = ""

    @staticmethod
    def _copy(frame):
        return JointFrame(frame.sequence, frame.angles.copy(), frame.captured_at)

    def submit_joints(self, angles, *, captured_at=None):
        """可从识别线程调用。角度单位为度；captured_at 是 monotonic 时间。"""
        try:
            # Copy before validation so caller mutations cannot alter an accepted frame.
            if not isinstance(angles, dict):
                raise ValueError("关节参数必须是角度字典")
            angles = angles.copy()
            self.config.targets(angles)
            now = time.monotonic()
            captured_at = now if captured_at is None else captured_at
            if (type(captured_at) not in (int, float) or not math.isfinite(captured_at)
                    or captured_at > now or now - captured_at > self.max_age):
                raise ValueError("视觉帧时间无效或已过期；请使用 time.monotonic() 的采集时间")
        except (ValueError, TypeError, KeyError) as exc:
            self.invalidate("拒绝视觉帧：" + str(exc))
            raise
        with self._condition:
            if time.monotonic() - captured_at > self.max_age:
                self.invalidate("视觉帧已过期")
                raise ValueError("视觉帧已过期")
            if self._frame is not None and captured_at < self._frame.captured_at:
                self.invalidate("视觉帧时间倒退")
                raise ValueError("视觉帧时间倒退")
            self._sequence += 1
            self._frame = JointFrame(self._sequence, angles, captured_at)
            self._condition.notify_all()
            return self._sequence

    def invalidate(self, reason="视觉跟踪丢失"):
        with self._condition:
            self._failure += 1
            self._reason = str(reason)
            self._frame = None
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            if self._frame is None or time.monotonic() - self._frame.captured_at > self.max_age:
                return None
            return self._copy(self._frame)

    def begin(self):
        with self._condition:
            if self.snapshot() is None:
                raise TimeoutError("尚无新鲜的关节参数，请先由视觉端传入有效帧")
            return self._failure

    def failure_since(self, epoch):
        with self._condition:
            return self._reason if self._failure != epoch else None

    def next_frame(self, sequence, epoch, cancel):
        with self._condition:
            while not cancel.is_set():
                failure = self.failure_since(epoch)
                if failure is not None:
                    raise TrackingLostError(failure)
                frame = self.snapshot()
                if frame is None:
                    raise FrameExpiredError("视觉关节参数已超时；本次联动停止，请重新开始")
                if frame.sequence != sequence:
                    return frame
                self._condition.wait(min(0.05, self.max_age))
        return None


class _FrameGuard:
    """将取消、输入故障及帧截止时间传到现有串口操作的每个取消检查点。"""

    def __init__(self, source, epoch, cancel):
        self.source, self.epoch, self.cancel = source, epoch, cancel
        self.deadline = None
        self.error = None

    def is_set(self):
        if self.cancel.is_set():
            return True
        reason = self.source.failure_since(self.epoch)
        if self.error is None and reason is not None:
            self.error = TrackingLostError(reason)
        if self.error is None and self.deadline is None and self.source.snapshot() is None:
            self.error = FrameExpiredError("初始化期间视觉参数已超时；本次联动停止")
        if self.error is None and self.deadline is not None and time.monotonic() >= self.deadline:
            self.error = FrameExpiredError("当前视觉帧已过期；本次联动停止，请重新开始")
        return self.error is not None

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.cancel.wait(min(remaining, 0.02))
        return True

    def check(self):
        self.is_set()
        if self.error is not None:
            raise self.error


def stream_joint_input(config, bus, source, *, stop_event, epoch=None, on_frame=None,
                       home_pose=None, emergency=None, notice=None):
    """Follow latest input; return only on controlled stop or visual loss."""
    if home_pose is not None:
        home_pose.targets(config)  # Validate before preparation or any motion.
    try:
        _stream_joint_input(config, bus, source, stop_event=stop_event, epoch=epoch, on_frame=on_frame)
    except (TrackingLostError, FrameExpiredError) as exc:
        if home_pose is None:
            raise
        if notice is not None:
            notice(str(exc) + "；结束跟随并尝试默认归位")
    if home_pose is not None:
        return return_to_home(config, bus, home_pose,
            emergency=emergency if emergency is not None else threading.Event(),
            on_frame=on_frame, notice=notice)


def _stream_joint_input(config, bus, source, *, stop_event, epoch=None, on_frame=None):
    """在调用方的 serial_bus 上运行；退出后的卸载由该上下文统一完成。"""
    if epoch is None:
        epoch = source.begin()
    guard = _FrameGuard(source, epoch, stop_event)
    session = CommandSession(config, bus, guard, nearby_start=True)
    # No movement is sent during preparation. Keep the source live, then select
    # the newest frame after preparation instead of executing an old first pose.
    prepared = session.prepare()
    guard.check()
    if not prepared or stop_event.is_set():
        return
    sequence = 0
    while not stop_event.is_set():
        frame = source.next_frame(sequence, epoch, stop_event)
        if frame is None:
            return
        guard.deadline = frame.captured_at + source.max_age
        reply = session.set_joint_positions(frame.angles)
        guard.check()
        if reply is None or stop_event.is_set():
            return
        sequence = frame.sequence
        if on_frame is not None:
            on_frame(frame.angles, reply)


class EmbeddedHandController:
    """供 OpenCV、Qt 等任意视觉程序调用的控制模块，不创建界面或 TCP 服务。"""

    def __init__(self, config, *, port="", baud=1000000, max_age=0.5, on_frame=None, home_pose=None):
        if isinstance(config, (str, Path)):
            config = HandConfig(config)
        self.config = HandConfig.from_data(config.to_data())
        self.input = JointInput(self.config, max_age=max_age)
        self.port, self.baud = port, baud
        self.on_frame = on_frame
        if home_pose is not None:
            home_pose.targets(self.config)
        self.home_pose = home_pose
        self._home_abort = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._thread = None
        self._closed = False
        self._state = "idle"
        self._error = None
        self._last_frame = None

    def submit_joints(self, angles, *, captured_at=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("控制模块已关闭")
            return self.input.submit_joints(angles, captured_at=captured_at)

    def tracking_lost(self, reason="视觉跟踪丢失"):
        self.input.invalidate(reason)

    def start(self, *, enable_motion=False):
        """显式授权实机联动；首帧必须接近当前位置，之后只消费新帧。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("控制模块已关闭")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("已有实机操作正在运行或卸载")
            if enable_motion is not True:
                raise ValueError("实机联动需要显式设置 enable_motion=True")
            if not self.config.calibrated or not self.config.motors:
                raise ValueError("请先建立非空关节绑定并完成零位标定")
            integer(self.config.model_number, 1, 65535, "model_number")
            if not isinstance(self.port, str) or not self.port.strip():
                raise ValueError("请设置 FE-URT-2 的串口")
            integer(self.baud, 1, 4000000, "baud")
            epoch = self.input.begin()
            self._cancel = threading.Event()
            self._home_abort.clear()
            self._error, self._last_frame = None, None
            self._state = "running"
            self._thread = threading.Thread(target=self._run, args=(epoch,), name="embedded-hand")
            self._thread.start()

    def _record(self, angles, reply):
        with self._lock:
            self._last_frame = deepcopy({"joints_deg": angles, **reply})
        if self.on_frame is not None:
            self.on_frame(angles, reply)

    def _run(self, epoch):
        try:
            with serial_bus(self.port.strip(), self.baud) as bus:
                stream_joint_input(self.config, bus, self.input, stop_event=self._cancel,
                                   epoch=epoch, on_frame=self._record,
                                   home_pose=self.home_pose, emergency=self._home_abort)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._state = "error"
        else:
            with self._lock:
                self._state = "stopped"

    def status(self):
        """可从宿主界面轮询；running=False 之前，串口卸载已执行完毕。"""
        with self._lock:
            return {"state": self._state, "error": self._error,
                    "closed": self._closed,
                    "running": self._thread is not None and self._thread.is_alive(),
                    "last_frame": deepcopy(self._last_frame)}

    def stop(self):
        self._cancel.set()

    def emergency_stop(self):
        self._home_abort.set()
        self._cancel.set()

    def wait(self, timeout=None):
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def close(self, timeout=None):
        with self._lock:
            self._closed = True
            self.stop()
        if not self.wait(timeout):
            raise TimeoutError("实机操作仍在结束或卸载中，请继续等待")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
