#!/usr/bin/env python3
"""
Bionic Dexterous Hand - Main Application.

Real-time control of a bionic hand using MediaPipe hand tracking.
Maps detected hand joint angles to servo motors via serial communication
(or simulated mode for development without hardware).

Run:
    python -m src.main
    python -m src.main --sim       # Simulated mode (no hardware)
    python -m src.main --real      # Real hardware mode
    python -m src.main --config path/to/config.yaml
"""

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .hand_tracking.mediapipe_tracker import HandTracker, HandLandmarks
from .hand_mapper.joint_mapper import JointMapper, JointAngles
from .hand_mapper.gestures import GestureRecognizer, finger_state
from .hand_mapper.orientation import OrientationDetector
from .hand_mapper.motion_gestures import MotionRecognizer
from .motor_control.servo_manager import ServoManager
from .motor_control.serial_controller import (
    SerialController,
    RealSerial,
    SimulatedSerial,
    parse_ack,
)
from .ui.display import HandDisplay
from .ui.unity_bridge import UnityPoseStream


# ---------------------------------------------------------------------------
# Hardware manifest helpers
# ---------------------------------------------------------------------------

def _load_active_axis_specs(config: dict) -> dict:
    """Load active axis specifications when a hardware manifest is configured."""
    manifest_path = config.get("hardware_manifest")
    if not manifest_path:
        return {}

    path = Path(manifest_path)
    if not path.is_absolute():
        # Resolve against the config file's directory first.  The repository
        # root is a fallback so an existing root-relative path keeps working;
        # without it a config that moved would silently lose its limits.
        base = Path(config.get("_config_dir") or Path(__file__).parent.parent)
        candidate = base / path
        if not candidate.exists():
            root_candidate = Path(__file__).parent.parent / path
            if root_candidate.exists():
                candidate = root_candidate
        path = candidate
    try:
        with open(path, "r", encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to load hardware manifest {path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"Hardware manifest must be a mapping: {path}")

    raw_axes = manifest.get("axes", [])
    if isinstance(raw_axes, dict):
        raw_axes = [dict(spec, name=name) if isinstance(spec, dict) else {"name": name}
                    for name, spec in raw_axes.items()]
    if not isinstance(raw_axes, list):
        raise ValueError(f"Hardware manifest axes must be a list or mapping: {path}")

    return {
        axis.get("name"): axis
        for axis in raw_axes
        if isinstance(axis, dict)
        and axis.get("role", "active") == "active"
        and axis.get("name")
    }


def _apply_manifest_servo_config(
    manager: ServoManager,
    axis_specs: dict,
    *,
    full_range: bool = False,
    require_limits: bool = False,
) -> None:
    """Apply neutral positions and conservative manifest limits to a manager.

    With ``require_limits`` every servo must get a usable window from the
    manifest.  A real-hardware run must not silently fall back to the 0-180
    defaults: an axis with no recorded limit would be driven through a range
    nobody has cleared on the bench.
    """
    unconfigured: List[str] = []
    for name, servo in manager._servos.items():
        spec = axis_specs.get(name, {})
        try:
            servo.config.default_angle = float(
                spec.get("neutral_angle", servo.config.default_angle)
            )
        except (TypeError, ValueError):
            pass

        if not full_range:
            limits = spec.get("soft_limits", {})
        else:
            limits = spec.get("nominal_limits", {})
        applied = False
        if isinstance(limits, dict):
            try:
                lower = float(limits["min"])
                upper = float(limits["max"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if lower <= upper:
                    servo.config.min_angle = lower
                    servo.config.max_angle = upper
                    servo.config.default_angle = max(
                        lower, min(upper, servo.config.default_angle)
                    )
                    applied = True
        if require_limits and not applied:
            unconfigured.append(name)

        # ServoManager is constructed with its legacy 90° defaults before the
        # manifest is applied.  Initialize both runtime angles as well as the
        # config default so the first transmitted pose matches the manifest.
        servo.current_angle = servo.config.default_angle
        servo.target_angle = servo.config.default_angle
        servo.last_command_time = time.monotonic()

    if unconfigured:
        raise ValueError(
            "Real-hardware mode requires a limit window for every axis; "
            f"missing or invalid for: {', '.join(sorted(unconfigured))}"
        )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure the logging system."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "hand_config.yaml"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    The directory the config was read from is recorded under ``_config_dir`` so
    that a relative ``hardware_manifest`` path is resolved against the config
    file's own directory rather than the repository root.  A config kept next
    to its manifest keeps working when passed with ``--config``.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict):
        config["_config_dir"] = str(Path(config_path).resolve().parent)
    return config


# ---------------------------------------------------------------------------
# Bionic Hand Controller
# ---------------------------------------------------------------------------

class BionicHandController:
    """
    Main controller orchestrating the bionic hand system.

    Pipeline:
      Camera -> MediaPipe Hand Tracking -> Joint Angle Mapping -> Servo Control

    Components:
      - HandTracker:   MediaPipe hand landmark detection
      - JointMapper:   Landmark -> joint angle conversion
      - ServoManager:  Angle -> servo pulse management
      - SerialController: Hardware communication (or simulation)
      - HandDisplay:   Real-time visualization
    """

    # Maximum sequence id the firmware accepts in a "#<seq> " prefix.
    MAX_SEQUENCE_ID = 999

    # A confirmed-ACK timeout for one real-time frame.  Longer than a healthy
    # round trip at 115200 baud, short enough that a dead link is detected
    # within a few frames instead of stalling the vision loop.
    FRAME_ACK_TIMEOUT = 0.1

    # Consecutive unacknowledged frames tolerated before the controller stops.
    # One dropped line should not end a demo; a dead actuator path must.
    MAX_CONSECUTIVE_ACK_FAILURES = 3

    def __init__(self, config: dict, *, full_range: bool = False):
        """
        Initialize all subsystems from configuration.

        Args:
            config: Configuration dictionary loaded from YAML.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        self._paused = False
        self._shutdown_complete = False
        self._display_enabled = bool(config.get("display", {}).get("enabled", True))

        # --- Hand Tracker ---
        mp_cfg = config.get("mediapipe", {})
        self.tracker = HandTracker(
            static_image_mode=mp_cfg.get("static_image_mode", False),
            max_num_hands=mp_cfg.get("max_num_hands", 1),
            min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.7),
            min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
        )

        # --- Joint Mapper ---
        smoothing_cfg = config.get("servos", {}).get("smoothing", {})
        # MediaPipe's handedness label assumes a mirrored camera feed, and it
        # is wrong often enough that wrist_rot sat pinned at 0 for an entire
        # measured capture.  The orientation block already carries an override,
        # so the mapper reads the same knob rather than growing a second
        # setting that could contradict it.
        forced_handedness = (
            str((config.get("orientation", {}) or {})
                .get("forced_handedness", "") or "").strip()
            or None
        )
        self.joint_mapper = JointMapper(
            smoothing_enabled=smoothing_cfg.get("enabled", True),
            smoothing_alpha=smoothing_cfg.get("alpha", 0.3),
            max_step=smoothing_cfg.get("max_step", 5.0),
            forced_handedness=forced_handedness,
        )

        # --- Servo Manager ---
        pin_assignments = config.get("servos", {}).get("pins", {})
        self.servo_manager = ServoManager(pin_assignments)
        is_simulation = config.get("simulation", {}).get("enabled", True)
        if not is_simulation and not config.get("hardware_manifest"):
            raise ValueError(
                "Real-hardware mode requires 'hardware_manifest' in the config: "
                "without it every axis would run on unverified default limits"
            )
        self._axis_specs = _load_active_axis_specs(config)
        _apply_manifest_servo_config(
            self.servo_manager,
            self._axis_specs,
            full_range=full_range,
            require_limits=not is_simulation,
        )

        # --- Serial Controller ---
        is_sim = is_simulation
        if is_sim:
            self.logger.info("Running in SIMULATION mode (no hardware)")
            self.serial = SerialController(SimulatedSerial(verbose=False))
        else:
            serial_cfg = config.get("serial", {})
            self.logger.info(
                "Running in HARDWARE mode on %s", serial_cfg.get("port", "COM3")
            )
            self.serial = SerialController(
                RealSerial(
                    port=serial_cfg.get("port", "COM3"),
                    baudrate=serial_cfg.get("baudrate", 115200),
                    timeout=serial_cfg.get("timeout", 0.1),
                )
            )

        # --- Camera ---
        cam_cfg = config.get("camera", {})
        self.camera_index = cam_cfg.get("index", 0)
        self.camera_width = cam_cfg.get("width", 640)
        self.camera_height = cam_cfg.get("height", 480)
        self.camera_fps = cam_cfg.get("fps", 30)
        self.flip_horizontal = cam_cfg.get("flip_horizontal", False)
        self._cap: Optional[cv2.VideoCapture] = None

        # --- Display ---
        self.display = HandDisplay(
            show_camera=self._display_enabled,
            show_dashboard=self._display_enabled,
            window_name="Bionic Hand Control",
        )

        # --- Unity pose stream ---
        # Optional, off unless configured: it is a second consumer of the same
        # angles, and the system must run identically without it.
        unity_cfg = config.get("unity", {}) or {}
        self._unity_enabled = bool(unity_cfg.get("enabled", False))
        self.unity_stream = UnityPoseStream(
            host=str(unity_cfg.get("host", "127.0.0.1")),
            port=int(unity_cfg.get("port", 8888)),
            enabled=self._unity_enabled,
        )

        # --- Gesture and orientation ---
        # Both are stateful (debounce and hysteresis respectively), so one
        # instance each is kept for the life of the controller.  Rebuilding them
        # per frame would throw away exactly the history that stops the readout
        # flickering.
        self.gesture_recognizer = GestureRecognizer()
        # Preset poses for each gesture name (from config).  In gesture-driven
        # mode, when a gesture is recognised the hand is driven to this pose
        # instead of to the raw per-joint angles.  This makes the motion much
        # crisper and less noisy -- the raw tracking carries a couple of
        # degrees of jitter per joint, and the presets are what the user
        # actually wants the hand to do.
        self._gesture_presets: Dict[str, Dict[str, float]] = {
            name: dict(info.get("angles", {}))
            for name, info in (config.get("gestures") or {}).items()
            if isinstance(info, dict) and "angles" in info
        }
        # Gesture-driven mode: use preset poses instead of raw joint angles.
        # Smoother, less jitter, but only works for the gestures that have a
        # preset definition.
        self._gesture_driven = bool(
            (config.get("gestures") or {}).get("gesture_driven", False)
        )
        # Gesture assist: every finger is independently blended toward a
        # canonical extended or flexed target, weighted by how far the finger
        # is past the recognition threshold.  This denoises each finger on its
        # own timetable -- a clear index finger rides a clean target while a
        # wobbly ring finger stays live -- and it works without having to match
        # one specific named gesture.
        self._gesture_assist = bool(
            (config.get("gestures") or {}).get("gesture_assist", False)
        )
        self._assist_strength = float(
            (config.get("gestures") or {}).get("assist_strength", 0.7)
        )
        # Canonical PIP values for the assist target.  These define "fully
        # extended" and "fully flexed" in visual terms; they don't have to
        # match any specific gesture exactly.
        self._assist_extended_pip = 172.0
        self._assist_flexed_pip = 25.0
        # Easing toward the assisted target, in degrees per second.
        self._gesture_drive_speed = float(
            (config.get("gestures") or {}).get("drive_speed_dps", 300.0)
        )
        # The eased pose for gesture-driven and gesture-assist modes.
        self._gesture_pose: Optional[Dict[str, float]] = None
        self._gesture_pose_target: Optional[Dict[str, float]] = None
        # EMA time constant for the assisted pose, in seconds.  A small value
        # tracks the live signal (jittery); a larger value smooths out noise
        # but adds lag.  Set with gestures.assist_smooth_tau.
        self._assist_smooth_tau = float(
            (config.get("gestures") or {}).get("assist_smooth_tau", 0.08)
        )
        self._assist_smoothed: Optional[Dict[str, float]] = None
        # Front/back is decided from the sign of the palm's projected area, and
        # that sign flips with the hand.  A misclassified left/right would
        # silently invert the answer, so the demo pins the handedness unless
        # the config says otherwise.
        orientation_cfg = config.get("orientation", {}) or {}
        forced = str(orientation_cfg.get("forced_handedness", "") or "").strip() or None
        self.motion_recognizer = MotionRecognizer()
        self._last_motion = None
        self.orientation_detector = OrientationDetector(
            invert_front_back=bool(orientation_cfg.get("invert_front_back", False)),
            forced_handedness=forced,
        )
        self._last_gesture = None
        self._last_orientation = None

        # --- State ---
        self._last_landmarks: Optional[HandLandmarks] = None
        self._last_angles: Optional[JointAngles] = None
        self._frame_interval = 1.0 / max(self.camera_fps, 1)
        self._last_health_check = 0.0
        self._serial_error_count = 0
        self._serial_ack_count = 0
        # Protocol sequence id, used to correlate an acknowledgment with the
        # exact frame it answers.  Without it a delayed ACK from an earlier
        # frame is indistinguishable from the current one.
        self._serial_seq = 0
        self._consecutive_ack_failures = 0
        self._last_successful_output_time = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Initialize all subsystems and start the control loop."""

        self.logger.info("=" * 50)
        self.logger.info("Bionic Dexterous Hand Controller v1.0.0")
        self.logger.info("=" * 50)

        # 1. Start MediaPipe tracker
        try:
            self.tracker.start()
            self.logger.info("[OK] Hand tracker started (model=%s, mode=%s)",
                           self.tracker.model_path,
                           "IMAGE" if self.tracker.static_image_mode else "VIDEO")
        except Exception as e:
            self.logger.error("Failed to start hand tracker: %s", e)
            return False

        # 2. Open camera
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        if not self._cap.isOpened():
            self.logger.error("Failed to open camera (index=%d)", self.camera_index)
            self.tracker.stop()
            return False

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.logger.info("[OK] Camera opened: %dx%d", int(actual_w), int(actual_h))

        # 3. Connect to hardware (or simulation).  A controller that was
        # explicitly configured for real hardware must not silently degrade to
        # display-only operation: continuing without a verified link can make
        # the UI look healthy while the actuator path is dead.
        if not self.serial.connect():
            self.logger.error("Could not connect to the configured serial backend")
            self.stop()
            return False

        # 4. Open display only when GUI output is enabled.
        if self._display_enabled:
            self.display.open()

        # 4b. Start the Unity pose stream.  Its failure is never fatal: a Unity
        # client that is not running must not stop the hand from working.
        if self._unity_enabled:
            if self.unity_stream.start():
                self.logger.info("[OK] Unity pose stream started")
            else:
                self.logger.warning(
                    "Unity pose stream requested but unavailable; continuing without it"
                )

        # 5. Set initial pose and require a complete acknowledgment before
        # entering the control loop.  This also validates the firmware's
        # driver-ready state when using a real serial connection.
        self.servo_manager.set_default_pose()
        if not self._send_current_pose(require_ack=True):
            self.logger.error("Initial pose was not acknowledged; refusing to run")
            self.stop()
            return False
        self.serial.enable_auto_reconnect()

        self._running = True
        self.logger.info("[OK] Controller started. Press 'q' or ESC to quit.")
        return True

    def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self.logger.info("Shutting down...")
        self._running = False

        try:
            # Reset servos to default position before disconnecting.  Interpolate
            # and confirm each packet so shutdown does not transmit the old pose.
            self.servo_manager.set_default_pose()
            deadline = time.perf_counter() + 10.0
            while time.perf_counter() < deadline:
                self.servo_manager.update_interpolation(0.02)
                if not self._send_current_pose(require_ack=True):
                    break
                current = self.servo_manager.get_all_current_angles()
                target = self.servo_manager.get_all_target_angles()
                if all(
                    abs(current[name] - target[name]) < 0.5
                    for name in target
                ):
                    break
                time.sleep(0.02)
        finally:
            self.serial.disconnect()
            time.sleep(0.05)

            # Stop subsystems
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                self.logger.info("[OK] Camera released")

            self.tracker.stop()
            self.logger.info("[OK] Hand tracker stopped")

            self.logger.info(
                "Serial stats: failed_frames=%d, acknowledged_frames=%d",
                self._serial_error_count,
                self._serial_ack_count,
            )

            self.display.close()
            self.logger.info("[OK] Display closed")

            if self.unity_stream.is_running:
                self.logger.info(
                    "Unity stream stats: packets_sent=%d", self.unity_stream.packets_sent
                )
            self.unity_stream.close()

            cv2.destroyAllWindows()
            self.logger.info("Bionic Hand Controller stopped.")

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main control loop. Blocks until user quits."""
        if not self._running:
            raise RuntimeError("Controller not started. Call start() first.")

        last_frame_time = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            # ------------------------------------------------------------------
            # 1. Capture frame
            # ------------------------------------------------------------------
            ret, frame = self._cap.read()
            if not ret:
                self.logger.warning("Failed to read camera frame")
                if self._display_enabled:
                    self._handle_keyboard(cv2.waitKey(1) & 0xFF)
                time.sleep(0.01)
                continue

            # Resize to configured resolution (camera may ignore our requested size)
            frame = cv2.resize(frame, (self.camera_width, self.camera_height))

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # ------------------------------------------------------------------
            # 2. Hand tracking
            # ------------------------------------------------------------------
            landmarks = self.tracker.process(frame)

            # ------------------------------------------------------------------
            # 3. Joint angle mapping
            # ------------------------------------------------------------------
            angles = None
            if landmarks is not None:
                angles = self.joint_mapper.compute_angles(landmarks)

            # The pose sent to consumers (Unity, serial, UI).  In gesture-driven
            # mode this is the eased preset pose -- much cleaner than the raw
            # per-joint tracking, which is jittery especially on the ring/pinky
            # fingers where MediaPipe's keypoint noise is highest.
            display_pose: Optional[Dict[str, float]] = None
            if angles is not None:
                display_pose = angles.to_dict()

            # ------------------------------------------------------------------
            # 3b. Gesture, orientation, motion
            # ------------------------------------------------------------------
            # Stateful recognisers: must see every frame in order.  Computed
            # before motor control so gesture-driven mode can use the result
            # to pick a target pose.
            recognition_angles = (
                self.joint_mapper.instant_angles if angles is not None else None
            )
            gesture = self.gesture_recognizer.update(recognition_angles, landmarks)
            orientation = self.orientation_detector.update(landmarks)
            motion = self.motion_recognizer.update(landmarks)
            self._last_motion = motion
            self._last_gesture = gesture
            self._last_orientation = orientation

            # ------------------------------------------------------------------
            # 4. Motor control
            # ------------------------------------------------------------------
            if time.perf_counter() - self._last_health_check >= 1.0:
                self._last_health_check = time.perf_counter()
                if not self.serial.heartbeat():
                    self.logger.warning("Serial heartbeat failed")

            if not self._paused:
                # Decide the target pose.
                target = display_pose

                if self._gesture_assist and angles is not None:
                    # Per-finger gesture assist: each finger is independently
                    # blended toward a canonical extended or flexed PIP value,
                    # weighted by how decisively it is in that state.
                    # A finger that is clearly extended rides a clean extended
                    # target (no jitter), and a finger in the ambiguous zone
                    # stays live.  This gives most of gesture_driven's stability
                    # without needing a named gesture to match first, so partial
                    # and in-between gestures still look alive.
                    from .hand_mapper.gestures import (
                        per_finger_margins, EXTENDED, FLEXED,
                    )
                    margins = per_finger_margins(
                        self.joint_mapper.instant_angles or angles
                    )
                    assisted = dict(display_pose)
                    strength = max(0.0, min(1.0, self._assist_strength))

                    for finger, m in margins.items():
                        ext = m.get(EXTENDED)
                        flx = m.get(FLEXED)
                        weight = 0.0
                        target_val = None
                        if ext is not None and ext > 0:
                            weight = ext * strength
                            target_val = self._assist_extended_pip
                        elif flx is not None and flx > 0:
                            weight = flx * strength
                            target_val = self._assist_flexed_pip

                        if target_val is None or weight <= 0:
                            continue

                        # Apply to the PIP axis of this finger.  The MCP is
                        # left alone -- its measured range is too small for the
                        # assist to meaningfully help, and forcing it would look
                        # stiff.
                        pip_key = f"{finger}_pip"
                        if pip_key in assisted:
                            cur = assisted[pip_key]
                            assisted[pip_key] = cur + (target_val - cur) * weight

                    # EMA-smooth the assisted pose before sending it out.
                    # The per-finger blend produces a clean target but it still
                    # updates every frame in lockstep with MediaPipe's per-frame
                    # noise; smoothing the result in time removes the remaining
                    # jitter without adding perceptible lag.
                    dt_s = min(loop_start - last_frame_time, 0.1)
                    alpha = 1.0 if dt_s <= 0 or self._assist_smooth_tau <= 0 else (
                        1.0 - math.exp(-dt_s / self._assist_smooth_tau)
                    )
                    if self._assist_smoothed is None:
                        self._assist_smoothed = dict(assisted)
                    else:
                        for k, v in assisted.items():
                            prev = self._assist_smoothed.get(k, v)
                            self._assist_smoothed[k] = prev + (v - prev) * alpha
                    target = self._assist_smoothed
                    display_pose = self._assist_smoothed

                elif self._gesture_assist:
                    # A dropped frame in assist mode.  There is nothing new to
                    # blend, so the smoothed pose stays as it is and the servos
                    # keep their last target.  Falling through to the branches
                    # below reset the filter, so the next detection started over
                    # at the unsmoothed pose -- a jump on every lost frame.  With
                    # gesture_driven also on and the recogniser still holding a
                    # preset gesture across the gap, it went on to call
                    # angles.to_dict() on None and the program exited.
                    pass

                elif (self._gesture_driven and gesture.matched
                        and gesture.name in self._gesture_presets
                        # Seeding the eased pose needs a detected frame.
                        and (angles is not None or self._gesture_pose is not None)):
                    # Classic gesture-driven: snap to the full preset pose when
                    # a named gesture matches.
                    preset = self._gesture_presets[gesture.name]
                    if self._gesture_pose is None:
                        self._gesture_pose = dict(angles.to_dict())
                    self._gesture_pose_target = preset
                    dt = min(loop_start - last_frame_time, 0.1)
                    max_step = self._gesture_drive_speed * dt
                    for k, v in preset.items():
                        cur = self._gesture_pose.get(k, 90.0)
                        diff = v - cur
                        if abs(diff) <= max_step:
                            self._gesture_pose[k] = v
                        else:
                            self._gesture_pose[k] = cur + max_step * (1 if diff > 0 else -1)
                    target = dict(self._gesture_pose)
                    display_pose = target
                else:
                    if self._gesture_pose is not None and angles is not None:
                        self._gesture_pose = dict(angles.to_dict())
                    self._assist_smoothed = None  # back to live tracking

                if target is not None:
                    self.servo_manager.set_target_angles(target)

                # Interpolation is time-based, so it has to run every frame.
                # Gating it on a fresh detection froze the hand for exactly the
                # frames MediaPipe dropped, and then resumed with the whole gap
                # still left to travel -- a freeze followed by a jump, several
                # times a second.
                dt = min(loop_start - last_frame_time, 0.1)
                self.servo_manager.update_interpolation(dt)

                command = self.servo_manager.get_serial_command()
                if not self._deliver_pose(command):
                    # The pose could not be confirmed.  Keep counting; once the
                    # limit is reached the loop stops instead of continuing to
                    # drive a vision pipeline whose commands reach nothing.
                    if self._consecutive_ack_failures >= self.MAX_CONSECUTIVE_ACK_FAILURES:
                        self.logger.error(
                            "Serial link lost: %d consecutive frames were not acknowledged",
                            self._consecutive_ack_failures,
                        )
                        self.servo_manager.request_safe_pose("serial link lost")
                        self._running = False
                        break

            # ------------------------------------------------------------------
            # 5. Visualization
            # ------------------------------------------------------------------

            # The Unity stream is fed independently of the OpenCV window: it is
            # a second consumer of the same angles, and a headless setup (only
            # Unity open) must work just as well as the debug window.
            if angles is not None and self.unity_stream.is_running:
                self.unity_stream.send(
                    display_pose,
                    gesture=gesture.name,
                    orientation=orientation.facing,
                    orientation_confidence=orientation.confidence,
                    motion=motion.name if motion.matched else None,
                )

            if self._display_enabled:
                self._draw_landmarks_on_frame(frame, landmarks)

                # Per-finger state for the dashboard -- so the operator can see
                # at a glance which finger is preventing a gesture from matching.
                finger_states = {}
                if recognition_angles is not None:
                    for f in ("thumb", "index", "middle", "ring", "pinky"):
                        finger_states[f] = finger_state(recognition_angles, f)

                self.display.update(
                    camera_frame=frame,
                    landmarks=landmarks,
                    angles=angles,
                    fps=self.tracker.fps,
                    gesture_name=gesture.name if gesture.matched else None,
                    finger_states=finger_states,
                )

                if not self.display.show(delay_ms=1):
                    self._running = False
                    break

                # HandDisplay.show() owns the single waitKey call.
                self._handle_keyboard(self.display.last_key)

            # ------------------------------------------------------------------
            # 6. Frame rate control
            # ------------------------------------------------------------------
            elapsed = time.perf_counter() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            last_frame_time = loop_start

        # End of loop - clean shutdown
        self.stop()

    # ------------------------------------------------------------------
    # Keyboard handler
    # ------------------------------------------------------------------

    def _handle_keyboard(self, key: int) -> None:
        """Handle keyboard shortcuts."""
        if key == -1:
            return

        key_char = chr(key) if 32 <= key < 127 else None

        if key_char == "p":
            self._paused = not self._paused
            state = "PAUSED" if self._paused else "RESUMED"
            self.logger.info("%s", state)

        elif key_char == "r":
            self.joint_mapper.reset_smoothing()
            self.servo_manager.set_default_pose()
            self.logger.info("Reset pose and smoothing")

        elif key_char == "d":
            self.servo_manager.set_default_pose()
            self.logger.info("Default pose")

        elif key_char == "h":
            self._print_help()

        elif key_char == "s":
            self._print_status()

    def _print_help(self) -> None:
        """Print keyboard shortcuts."""
        print("\n" + "=" * 40)
        print("Keyboard Shortcuts:")
        print("  p  - Pause/Resume motor control")
        print("  r  - Reset pose and smoothing")
        print("  d  - Set default (open) pose")
        print("  h  - Show this help")
        print("  s  - Print servo status")
        print("  q / ESC - Quit")
        print("=" * 40 + "\n")

    def _print_status(self) -> None:
        """Print current servo status."""
        print("\n" + self.servo_manager.get_state_summary() + "\n")

    def _send_current_pose(self, *, require_ack: bool = False) -> bool:
        """Send the current interpolated pose, optionally requiring a full ACK."""
        if not self.serial.is_connected:
            return False
        command = self.servo_manager.get_serial_command()
        if not require_ack:
            ok = self.serial.send_command(command)
            if not ok:
                self._serial_error_count += 1
            return ok

        return self._deliver_pose(command, timeout=0.5)

    def _next_sequence(self) -> int:
        """Return the next protocol sequence id, wrapping within the wire range."""
        self._serial_seq = (self._serial_seq + 1) % (self.MAX_SEQUENCE_ID + 1)
        return self._serial_seq

    @staticmethod
    def _count_fields(command: str) -> int:
        """Count the joint fields in one serial frame."""
        return sum(1 for field in command.strip().split(",") if ":" in field)

    def _deliver_pose(
        self, command: str, *, timeout: Optional[float] = None
    ) -> bool:
        """Send one pose frame and require the firmware to acknowledge it.

        Returns True only when the acknowledgment echoes our sequence id and
        reports every field in the frame.  A partial or mismatched ACK counts as
        a failure: the axes the firmware dropped are silently not being driven,
        so treating it as success would hide a partially dead hand.
        """
        expected = self._count_fields(command)
        response = self.serial.send_and_receive(
            command,
            timeout=self.FRAME_ACK_TIMEOUT if timeout is None else timeout,
            seq=self._next_sequence(),
        )
        ack = parse_ack(response)
        if ack is None or ack[1] != expected:
            self._serial_error_count += 1
            self._consecutive_ack_failures += 1
            self.servo_manager.note_output_failure()
            self.logger.warning(
                "Frame not acknowledged (%d fields expected): %r", expected, response
            )
            return False

        self._serial_ack_count += 1
        self._consecutive_ack_failures = 0
        self._last_successful_output_time = time.monotonic()
        self.servo_manager.note_output_acknowledged()
        return True

    def _draw_landmarks_on_frame(
        self, frame: np.ndarray, landmarks: Optional[HandLandmarks]
    ) -> None:
        """Draw detection info on the camera frame (in-place)."""
        if landmarks is None:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        # Handedness label
        cv2.putText(
            frame,
            f"{landmarks.handedness} hand  {landmarks.handedness_confidence:.1%}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        # Pause indicator
        if self._paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def install_signal_handlers(controller, signals=(signal.SIGINT, signal.SIGTERM)):
    """Route shutdown signals to a clean controller stop.

    Kept at module level so the shutdown path can be tested without starting
    the full application (camera, MediaPipe, serial port).
    """
    def signal_handler(sig, frame):
        logging.getLogger("main").info("Received signal %s, shutting down...", sig)
        controller.stop()
        raise SystemExit(0)

    for sig in signals:
        signal.signal(sig, signal_handler)
    return signal_handler


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Bionic Dexterous Hand Controller - MediaPipe + Motor Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                     # Run with default config (simulation mode)
  python -m src.main --real              # Run with real hardware
  python -m src.main --sim               # Explicit simulation mode
  python -m src.main --config my_hand.yaml
  python -m src.main --no-display        # Headless mode (no GUI)
  python -m src.main --camera 1          # Use camera index 1
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Force simulation mode (no hardware communication)",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Force real hardware mode (serial communication)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run in headless mode (no GUI windows)",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device index (overrides config)",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (overrides config)",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List available cameras and exit",
    )
    parser.add_argument(
        "--full-range", action="store_true",
        help="Use nominal configured limits instead of the manifest bench-test window",
    )
    parser.add_argument(
        "--unity", action="store_true",
        help="Force the Unity pose stream on (UDP to the 3D hand model)",
    )
    parser.add_argument(
        "--no-unity", action="store_true",
        help="Force the Unity pose stream off",
    )

    args = parser.parse_args()

    # List cameras mode
    if args.list_cameras:
        _list_cameras()
        return

    # Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    if args.sim:
        config.setdefault("simulation", {})["enabled"] = True
    elif args.real:
        config.setdefault("simulation", {})["enabled"] = False

    if args.camera is not None:
        config.setdefault("camera", {})["index"] = args.camera

    if args.no_display:
        config.setdefault("display", {})["enabled"] = False

    if args.unity:
        config.setdefault("unity", {})["enabled"] = True
    elif args.no_unity:
        config.setdefault("unity", {})["enabled"] = False

    # Setup logging
    log_cfg = config.get("logging", {})
    log_level = args.log_level or log_cfg.get("level", "INFO")
    log_file = log_cfg.get("file")
    setup_logging(level=log_level, log_file=log_file)

    logger = logging.getLogger("main")

    # Create and run controller
    controller = BionicHandController(config, full_range=args.full_range)

    # Handle graceful shutdown
    install_signal_handlers(controller)

    # Override display setting
    if args.no_display:
        controller.display.show_camera = False
        controller.display.show_dashboard = False

    # Start and run
    try:
        if controller.start():
            controller.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
    finally:
        controller.stop()


def _list_cameras() -> None:
    """List available camera devices."""
    print("Scanning for cameras...")
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append((i, w, h))
            cap.release()
        else:
            break

    if available:
        print(f"Found {len(available)} camera(s):")
        for idx, w, h in available:
            print(f"  Camera {idx}: {w}x{h}")
    else:
        print("No cameras found.")


if __name__ == "__main__":
    main()
