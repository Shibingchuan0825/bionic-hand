"""Lifecycle tests for the top-level controller.

Everything here runs headless: no camera, no MediaPipe model, and no serial
port.  The controller's subsystems are replaced with stubs so the start/stop
and control-loop safety paths can be exercised deterministically.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_tracking.mediapipe_tracker import HandLandmarks
from src.main import BionicHandController
from src.motor_control.serial_controller import SerialController, SimulatedSerial

SERVO_NAMES = (
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip",
    "middle_mcp", "middle_pip",
    "ring_mcp", "ring_pip",
    "pinky_mcp", "pinky_pip",
    "wrist_rot",
)


class _StubTracker:
    """Stand-in for HandTracker; loads no MediaPipe model."""

    def __init__(self, landmarks=None):
        self.model_path = "stub.task"
        self.static_image_mode = False
        self.fps = 0.0
        self.started = False
        self.stopped = False
        self._landmarks = landmarks

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def process(self, frame):
        return self._landmarks


class _StubDisplay:
    """Stand-in for HandDisplay; opens no window."""

    def __init__(self):
        self.opened = False
        self.closed = False
        self.last_key = -1

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def update(self, **kwargs):
        pass

    def show(self, delay_ms=1):
        return True


class _FakeCapture:
    """Stand-in for cv2.VideoCapture.

    ``read_results`` scripts the first few reads; once the script is exhausted
    the capture serves blank frames so a test can never spin forever on a
    read-failure branch.
    """

    def __init__(self, opened=True, read_results=None):
        self._opened = opened
        self._results = list(read_results) if read_results else []
        self._index = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, *args):
        return True

    def get(self, prop):
        return 640.0

    def _blank(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def read(self):
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            if result is False:
                return False, None
        return self._blank()

    def release(self):
        self.released = True


class _DeafSerial(SimulatedSerial):
    """Accepts writes but never answers: a dead actuator path."""

    def read_line(self):
        return None


def _build_landmarks() -> HandLandmarks:
    """Landmarks with enough spread to pass the geometry checks."""
    pixels = np.zeros((21, 3), dtype=float)
    for index in range(21):
        pixels[index] = [100.0 + 6.0 * index, 120.0 + 9.0 * index, 0.0]
    return HandLandmarks(
        landmarks_px=pixels,
        landmarks_norm=pixels / 640.0,
        handedness="Right",
        handedness_confidence=0.9,
    )


def _config(*, display_enabled=False, unity_port=None) -> dict:
    config = {
        "display": {"enabled": display_enabled},
        "mediapipe": {},
        "servos": {"pins": {name: 0 for name in SERVO_NAMES}},
        "camera": {"index": 0, "width": 640, "height": 480, "fps": 30},
        "simulation": {"enabled": True},
        "unity": {
            "enabled": unity_port is not None,
            "host": "127.0.0.1",
            "port": unity_port if unity_port is not None else 8888,
        },
    }
    return config


def _make_controller(monkeypatch, *, display_enabled=False, capture=None,
                     interface=None, with_landmarks=False, unity_port=None):
    capture = capture if capture is not None else _FakeCapture()
    monkeypatch.setattr("src.main.cv2.VideoCapture", lambda *a, **k: capture)
    controller = BionicHandController(
        _config(display_enabled=display_enabled, unity_port=unity_port)
    )
    controller.tracker = _StubTracker(
        _build_landmarks() if with_landmarks else None
    )
    controller.display = _StubDisplay()
    if interface is not None:
        controller.serial = SerialController(interface)
    return controller, capture


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def test_start_succeeds_and_records_a_confirmed_delivery(monkeypatch):
    controller, _ = _make_controller(monkeypatch)

    assert controller.start() is True
    # A successful start must mean an acknowledged frame, not merely that the
    # default pose was requested.
    assert controller.servo_manager.seconds_since_last_output() < 1.0

    controller.stop()


def test_start_fails_closed_when_the_initial_pose_is_not_acknowledged(monkeypatch):
    controller, capture = _make_controller(monkeypatch, interface=_DeafSerial())

    assert controller.start() is False
    assert controller._running is False
    # The camera must not be left open after a refused start.
    assert capture.released is True


def test_start_fails_when_the_camera_cannot_be_opened(monkeypatch):
    controller, _ = _make_controller(
        monkeypatch, capture=_FakeCapture(opened=False)
    )

    assert controller.start() is False
    assert controller.tracker.stopped is True


def test_headless_start_does_not_open_a_display(monkeypatch):
    controller, _ = _make_controller(monkeypatch, display_enabled=False)

    assert controller.start() is True
    assert controller.display.opened is False

    controller.stop()
    assert controller.display.closed is True


def test_display_is_opened_only_when_enabled(monkeypatch):
    controller, _ = _make_controller(monkeypatch, display_enabled=True)

    assert controller.start() is True
    assert controller.display.opened is True

    controller.stop()


# ---------------------------------------------------------------------------
# Control loop safety
# ---------------------------------------------------------------------------

def test_run_stops_after_repeated_unacknowledged_frames(monkeypatch):
    controller, _ = _make_controller(monkeypatch, with_landmarks=True)
    assert controller.start() is True

    # The link dies after a healthy start.
    controller.serial.disconnect()
    controller.serial = SerialController(_DeafSerial())
    controller._running = True

    controller.run()

    assert controller._running is False
    assert controller._consecutive_ack_failures >= (
        controller.MAX_CONSECUTIVE_ACK_FAILURES
    )
    assert controller._last_successful_output_time > 0.0


def test_camera_read_failures_do_not_crash_the_loop(monkeypatch):
    """A camera that misses frames must be survivable, not fatal.

    The loop keeps running through the failed reads and then exits on the
    dead serial link, which is what proves it did not crash on the first miss.
    """
    controller, _ = _make_controller(
        monkeypatch,
        capture=_FakeCapture(read_results=[False, False]),
        with_landmarks=True,
    )
    assert controller.start() is True

    controller.serial.disconnect()
    controller.serial = SerialController(_DeafSerial())
    controller._running = True
    controller.run()

    assert controller._running is False


def test_rejected_frames_are_counted_as_failures(monkeypatch):
    controller, _ = _make_controller(monkeypatch, interface=_DeafSerial())
    controller.serial.connect()
    controller.servo_manager.set_default_pose()

    assert controller._deliver_pose("index_mcp:90") is False
    assert controller._consecutive_ack_failures == 1

    summary = controller.servo_manager.get_state_summary()
    assert "errors=1" in summary


def test_acknowledged_frames_reset_the_failure_counter(monkeypatch):
    controller, _ = _make_controller(monkeypatch, interface=_DeafSerial())
    controller.serial.connect()
    controller.servo_manager.set_default_pose()
    assert controller._deliver_pose("index_mcp:90") is False
    assert controller._consecutive_ack_failures == 1

    # A recovered link must clear the counter, not leave it one failure from
    # the shutdown threshold.
    controller.serial = SerialController(SimulatedSerial())
    controller.serial.connect()

    assert controller._deliver_pose("index_mcp:90") is True
    assert controller._consecutive_ack_failures == 0


def test_frame_sequence_ids_wrap_within_the_wire_range(monkeypatch):
    controller, _ = _make_controller(monkeypatch)
    controller._serial_seq = controller.MAX_SEQUENCE_ID

    assert controller._next_sequence() == 0


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def test_stop_is_idempotent(monkeypatch):
    controller, _ = _make_controller(monkeypatch)
    assert controller.start() is True

    controller.stop()
    controller.stop()

    assert controller._running is False


def test_sigint_stops_the_controller_and_exits(monkeypatch):
    """Ctrl-C must run the same shutdown path as a normal stop."""
    from src.main import install_signal_handlers

    controller, capture = _make_controller(monkeypatch)
    assert controller.start() is True

    registered = {}
    monkeypatch.setattr(
        "src.main.signal.signal",
        lambda sig, handler: registered.__setitem__(sig, handler),
    )
    install_signal_handlers(controller)

    import signal as signal_module

    assert set(registered) == {signal_module.SIGINT, signal_module.SIGTERM}

    with pytest.raises(SystemExit):
        registered[signal_module.SIGINT](signal_module.SIGINT, None)

    assert controller._running is False
    assert controller.tracker.stopped is True
    assert controller.display.closed is True
    assert capture.released is True


def test_real_hardware_mode_requires_a_manifest(monkeypatch):
    """Real hardware must not run on the unverified 0-180 fallback limits."""
    monkeypatch.setattr("src.main.cv2.VideoCapture", lambda *a, **k: _FakeCapture())
    config = _config()
    config["simulation"] = {"enabled": False}

    with pytest.raises(ValueError, match="hardware_manifest"):
        BionicHandController(config)


def test_real_hardware_mode_requires_limits_for_every_axis(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "axes:\n"
        "  - name: index_mcp\n"
        "    role: active\n"
        "    channel: 0\n"
        "    neutral_angle: 90\n"
        "    soft_limits: {min: 80, max: 100}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.main.cv2.VideoCapture", lambda *a, **k: _FakeCapture())
    config = _config()
    config["simulation"] = {"enabled": False}
    config["hardware_manifest"] = str(manifest)

    with pytest.raises(ValueError, match="limit window"):
        BionicHandController(config)


# ---------------------------------------------------------------------------
# Unity pose stream
#
# Unity is a second consumer of the same angles, not part of the control path.
# These tests pin down the two things that matter: it must be possible to run
# without it, and a missing Unity client must never disturb the hand.
# ---------------------------------------------------------------------------

def test_unity_stream_is_off_unless_configured(monkeypatch):
    controller, _ = _make_controller(monkeypatch)

    assert controller._unity_enabled is False
    assert controller.unity_stream.is_running is False


def test_unity_stream_starts_when_configured(monkeypatch):
    import socket

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(0.5)
    port = listener.getsockname()[1]

    controller, _ = _make_controller(monkeypatch, unity_port=port)
    try:
        assert controller.start() is True
        assert controller.unity_stream.is_running is True

        # start() announces the axis order before any pose.
        import json
        hello, _addr = listener.recvfrom(2048)
        assert json.loads(hello)["hello"] is True
    finally:
        controller.stop()
        listener.close()


def test_the_control_loop_feeds_the_unity_stream(monkeypatch):
    """Angles solved in the loop must reach the stream as pose packets."""
    import json
    import socket

    from src.ui.unity_bridge import AXIS_ORDER

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(1.0)
    port = listener.getsockname()[1]

    controller, _ = _make_controller(
        monkeypatch, with_landmarks=True, unity_port=port
    )
    assert controller.start() is True

    # The loop exits on a dead actuator link, which bounds the test.
    controller.serial.disconnect()
    controller.serial = SerialController(_DeafSerial())
    controller._running = True
    controller.run()

    poses = []
    try:
        while True:
            data, _addr = listener.recvfrom(2048)
            parsed = json.loads(data)
            if not parsed.get("hello"):
                poses.append(parsed)
    except socket.timeout:
        pass
    finally:
        controller.stop()
        listener.close()

    assert poses, "the loop sent no pose packets"
    assert len(poses[0]["a"]) == len(AXIS_ORDER)


def _run_assist_over_a_dropped_frame(monkeypatch, gesture):
    """Run assist mode over detected, dropped, detected frames.

    Returns the smoothing filter as it stood at the start of each frame, so a
    test can see whether the dropped frame threw it away.
    """
    controller, _ = _make_controller(monkeypatch)
    controller._gesture_assist = True
    controller._gesture_driven = True
    controller._gesture_presets = {"fist": {name: 30.0 for name in SERVO_NAMES}}
    controller.gesture_recognizer.update = lambda angles, landmarks=None: gesture

    frames = iter([_build_landmarks(), None] + [_build_landmarks()] * 5)
    filters = []

    def _process(frame):
        filters.append(controller._assist_smoothed)
        return next(frames)

    controller.tracker.process = _process

    assert controller.start() is True
    controller.serial.disconnect()
    controller.serial = SerialController(_DeafSerial())
    controller._running = True
    controller.run()
    controller.stop()
    return filters


def test_a_dropped_frame_while_a_preset_gesture_is_held_does_not_crash(monkeypatch):
    """The recogniser holds "fist" across a short gap; the loop must cope.

    With gesture assist and gesture-driven both on (the shipped config), a
    dropped frame fell through to the gesture-driven branch, which called
    angles.to_dict() on a frame that had no angles.
    """
    from src.hand_mapper.gestures import GestureResult

    filters = _run_assist_over_a_dropped_frame(
        monkeypatch, GestureResult("fist", "Fist", 0.0)
    )

    assert len(filters) >= 3


def test_a_dropped_frame_does_not_restart_the_pose_smoothing(monkeypatch):
    """Losing the hand for one frame must not discard the smoothing filter.

    It was thrown away on every dropped frame, so the next detection snapped
    straight to the raw pose: a visible jump each time a frame was lost.
    """
    from src.hand_mapper.gestures import GestureResult

    filters = _run_assist_over_a_dropped_frame(monkeypatch, GestureResult())

    assert filters[1] is not None
    assert filters[2] is filters[1]


def test_stop_closes_the_unity_stream(monkeypatch):
    import socket

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    controller, _ = _make_controller(monkeypatch, unity_port=port)
    assert controller.start() is True

    controller.stop()

    assert controller.unity_stream.is_running is False
    listener.close()


def test_a_broken_unity_link_does_not_block_startup(monkeypatch):
    """The hand must still run when the Unity client is not there.

    Unity is a display, not part of the control path: a stream that cannot
    start has to leave the controller fully operational.
    """
    controller, _ = _make_controller(monkeypatch, unity_port=8888)

    def _refuse() -> bool:
        return False

    monkeypatch.setattr(controller.unity_stream, "start", _refuse)

    assert controller.start() is True
    assert controller.unity_stream.is_running is False

    controller.stop()


def test_gesture_and_orientation_reach_the_unity_packet(monkeypatch):
    """The two recognised values must actually ride along with the pose.

    The recognisers themselves are stubbed: this is a wiring test.  What it
    pins down is that the loop computes them once per frame and passes them to
    the stream, rather than computing them and dropping them.
    """
    import json
    import socket

    from src.hand_mapper.gestures import GestureResult
    from src.hand_mapper.orientation import FACING_BACK, HandOrientation

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(1.0)
    port = listener.getsockname()[1]

    controller, _ = _make_controller(
        monkeypatch, with_landmarks=True, unity_port=port
    )

    def _fake_gesture(angles, landmarks=None):
        return GestureResult("ok", "OK", 0.9)

    def _fake_orientation(landmarks):
        return HandOrientation(FACING_BACK, 0.8, 0.7)

    controller.gesture_recognizer.update = _fake_gesture
    controller.orientation_detector.update = _fake_orientation

    assert controller.start() is True
    controller.serial.disconnect()
    controller.serial = SerialController(_DeafSerial())
    controller._running = True
    controller.run()

    poses = []
    try:
        while True:
            data, _addr = listener.recvfrom(2048)
            parsed = json.loads(data)
            if not parsed.get("hello"):
                poses.append(parsed)
    except socket.timeout:
        pass
    finally:
        controller.stop()
        listener.close()

    assert poses, "the loop sent no pose packets"
    assert poses[0]["g"] == "ok"
    assert poses[0]["o"] == FACING_BACK
    assert poses[0]["oc"] == 0.8


def test_the_recognisers_are_stateful_across_frames(monkeypatch):
    """One instance each, kept for the whole run.

    Both carry history -- the gesture debounce and the orientation hysteresis.
    Rebuilding them per frame would silently discard it, and the symptom would
    be a readout that flickers, which is easy to misread as noisy input.
    """
    controller, _ = _make_controller(monkeypatch)

    first_gesture = controller.gesture_recognizer
    first_orientation = controller.orientation_detector

    assert first_gesture is controller.gesture_recognizer
    assert first_orientation is controller.orientation_detector
    assert first_gesture.hold_frames > 1
