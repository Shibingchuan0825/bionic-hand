"""
Servo Angle Manager.

Manages servo motor states, angle mapping, and safety constraints.
Provides an abstract interface for servo control regardless of communication
backend (serial, simulated, or future CAN/EtherCAT).

Key responsibilities:
  - Maintain current servo angles
  - Enforce angle limits and smooth interpolation
  - Map normalized hand angles to physical servo pulse widths
  - Track servo health (position feedback, temperature - future)
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ServoState(Enum):
    """Operational state of a servo."""
    OK = "ok"
    LIMIT = "limit"         # At angle limit
    DISCONNECTED = "disconnected"
    FAULT = "fault"


@dataclass
class ServoConfig:
    """Configuration for a single servo."""
    name: str               # Human-readable name
    pin: int                # Arduino PWM pin
    min_angle: float = 0.0
    max_angle: float = 180.0
    default_angle: float = 90.0
    min_pulse_us: int = 500    # PWM pulse for 0°
    max_pulse_us: int = 2500   # PWM pulse for 180°
    inverted: bool = False     # Reverse direction
    # Degrees per second.  The mapper already eases every angle, so this is a
    # safety ceiling on the actuator, not the thing that makes motion smooth.
    # At 60 the two limits compounded and the servos were the slower half of a
    # half-second lag; a hobby servo does roughly 300 deg/s unloaded, so 180
    # still binds before the hardware does.
    speed_limit: float = 180.0


@dataclass
class ServoStateInfo:
    """Runtime state tracking for a servo."""
    config: ServoConfig
    current_angle: float
    target_angle: float
    state: ServoState = ServoState.OK
    # When a target was last received from the vision pipeline.  This is
    # intent, not proof that anything reached the hardware.
    last_command_time: float = 0.0
    # When the controller last confirmed the firmware acknowledged a frame.
    # Zero means no frame has ever been confirmed delivered.
    last_output_time: float = 0.0
    error_count: int = 0


# ---------------------------------------------------------------------------
# Servo Manager
# ---------------------------------------------------------------------------

class ServoManager:
    """
    Manages a collection of servo motors for the bionic hand.

    Features:
      - Angle clamping to safe operational ranges
      - Per-servo inversion for mechanical orientation
      - Angle-to-pulse-width conversion for standard RC servos
      - Stale-command timeout detection
      - Interpolation smoothing

    Usage:
        manager = ServoManager(servo_configs)
        manager.set_target_angles(joint_angles_dict)
        pulses = manager.get_pulse_widths()
        # Send pulses to hardware...
    """

    # Standard servo pulse range
    DEFAULT_MIN_PULSE_US = 500
    DEFAULT_MAX_PULSE_US = 2500
    DEFAULT_ANGLE_RANGE = 180.0

    # Safety: if no command for this long, flag as stale
    STALE_TIMEOUT_SEC = 2.0

    def __init__(self, pin_assignments: Dict[str, int]):
        """
        Initialize servo manager with pin assignments.

        Args:
            pin_assignments: Dict mapping joint names to Arduino PWM pins.
        """
        self._servos: Dict[str, ServoStateInfo] = {}

        # Create default config for each joint
        for joint_name, pin in pin_assignments.items():
            config = ServoConfig(
                name=joint_name,
                pin=pin,
                min_angle=0.0,
                max_angle=180.0,
                default_angle=90.0,
            )
            self._servos[joint_name] = ServoStateInfo(
                config=config,
                current_angle=config.default_angle,
                target_angle=config.default_angle,
                last_command_time=time.monotonic(),
            )

        logger.info(
            "ServoManager initialized with %d servos: %s",
            len(self._servos),
            list(self._servos.keys()),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def servo_names(self) -> List[str]:
        return list(self._servos.keys())

    @property
    def servo_count(self) -> int:
        return len(self._servos)

    def get_current_angle(self, servo_name: str) -> float:
        """Get the current angle of a specific servo."""
        if servo_name not in self._servos:
            raise KeyError(f"Unknown servo: {servo_name}")
        return self._servos[servo_name].current_angle

    def get_all_current_angles(self) -> Dict[str, float]:
        """Get all current servo angles."""
        return {name: s.current_angle for name, s in self._servos.items()}

    def get_all_target_angles(self) -> Dict[str, float]:
        """Get all target servo angles."""
        return {name: s.target_angle for name, s in self._servos.items()}

    # ------------------------------------------------------------------
    # Command Interface
    # ------------------------------------------------------------------

    def set_target_angles(self, angles: Dict[str, float]) -> None:
        """
        Set target angles for all servos.

        This is the primary interface used each frame. Angles are clamped
        to safe ranges and stored as targets for gradual interpolation.

        Args:
            angles: Dict mapping joint names to target angles in degrees.
        """
        now = time.monotonic()

        for name, target in angles.items():
            if name not in self._servos:
                logger.warning("Unknown servo target: %s", name)
                continue

            servo = self._servos[name]
            config = servo.config

            if not isinstance(target, (int, float)) or not math.isfinite(target):
                logger.warning("Ignoring non-finite target for servo %s: %r", name, target)
                servo.state = ServoState.FAULT
                continue

            # Clamp to safe range
            clamped = max(config.min_angle, min(config.max_angle, target))

            # Apply inversion if configured
            if config.inverted:
                clamped = config.max_angle - (clamped - config.min_angle)

            servo.target_angle = clamped
            servo.last_command_time = now
            servo.state = ServoState.OK

    def set_single_angle(self, servo_name: str, angle: float) -> None:
        """Set target angle for a single servo."""
        self.set_target_angles({servo_name: angle})

    def set_default_pose(self) -> None:
        """Reset all servos to their default (relaxed) positions."""
        defaults = {
            name: servo.config.default_angle
            for name, servo in self._servos.items()
        }
        self.set_target_angles(defaults)
        logger.info("All servos set to default pose")

    def request_safe_pose(self, reason: str) -> None:
        """Drive every joint toward its default pose after a fault.

        This is a software recovery action, not an emergency stop.  It only
        moves *targets*: the servos still have to be commanded and the frames
        still have to reach the hardware.  If the actuator path is dead nothing
        moves, so this must never be presented as a safety function.  The only
        real emergency stop is cutting actuator power at the power
        distribution; see the Gate A/B hardware notes in TODO.md.
        """
        self.set_default_pose()
        logger.warning("SAFE POSE REQUESTED (%s): targets reset to default", reason)

    # ------------------------------------------------------------------
    # Delivery tracking
    # ------------------------------------------------------------------

    def note_output_acknowledged(self) -> None:
        """Record that the hardware acknowledged a frame carrying this pose.

        Only this method proves delivery.  ``set_target_angles()`` records
        intent and must not be used to claim the hardware is being driven.
        """
        now = time.monotonic()
        for servo in self._servos.values():
            servo.last_output_time = now

    def note_output_failure(self) -> None:
        """Record that a frame was not acknowledged by the hardware."""
        for servo in self._servos.values():
            servo.error_count += 1

    def seconds_since_last_output(self) -> float:
        """Return the age of the most recent confirmed delivery, in seconds.

        Returns ``inf`` when no frame has ever been confirmed, so callers that
        compare against a timeout fail closed rather than treating "never
        delivered" as "delivered just now".
        """
        latest = max(
            (servo.last_output_time for servo in self._servos.values()),
            default=0.0,
        )
        if latest <= 0.0:
            return float("inf")
        return time.monotonic() - latest

    # ------------------------------------------------------------------
    # Interpolation & Pulse Generation
    # ------------------------------------------------------------------

    def update_interpolation(self, dt: float) -> None:
        """
        Move current angles toward targets with rate-limited interpolation.

        Call this once per control loop iteration.

        Args:
            dt: Time delta since last update (seconds).
        """
        if not isinstance(dt, (int, float)) or not math.isfinite(dt):
            logger.warning("Ignoring non-finite interpolation delta: %r", dt)
            return
        dt = max(0.0, dt)

        for name, servo in self._servos.items():
            target = servo.target_angle
            current = servo.current_angle

            if abs(target - current) < 0.1:
                servo.current_angle = target
                continue

            # Rate-limited step
            config = servo.config
            if config.speed_limit > 0:
                max_delta = config.speed_limit * dt
            else:
                max_delta = float("inf")

            delta = target - current
            step = max(-max_delta, min(max_delta, delta))
            servo.current_angle += step

            # Flag if at limit
            if servo.current_angle <= config.min_angle or \
               servo.current_angle >= config.max_angle:
                servo.state = ServoState.LIMIT

    def _get_safe_output_angle(self, servo: ServoStateInfo) -> float:
        """Return a bounded angle, using neutral when the command is stale.

        "Stale" here means the *vision pipeline* stopped producing targets, the
        case this host-side rule can actually detect.  A dead serial link is not
        detected this way: the firmware's own command timeout returns the hand
        to its default pose when frames stop arriving.  Loss of acknowledgment
        is tracked separately in ``seconds_since_last_output()``.
        """
        config = servo.config
        angle = servo.current_angle

        # Apply stale-state protection to every output path.  The runtime uses
        # the serial angle protocol, so keeping this rule only in
        # get_pulse_widths() would leave that path unprotected.
        if time.monotonic() - servo.last_command_time > self.STALE_TIMEOUT_SEC:
            angle = config.default_angle
            servo.state = ServoState.DISCONNECTED

        return max(config.min_angle, min(config.max_angle, angle))

    def get_pulse_widths(self) -> Dict[str, int]:
        """
        Convert current servo angles to PWM pulse widths (microseconds).

        Standard RC servo mapping:
          500us  = 0°
          1500us = 90° (center)
          2500us = 180°

        Returns:
            Dict mapping joint names to pulse widths in microseconds.
        """
        pulses = {}
        for name, servo in self._servos.items():
            config = servo.config
            angle = self._get_safe_output_angle(servo)

            # Linear mapping: angle → pulse
            angle_ratio = angle / self.DEFAULT_ANGLE_RANGE
            pulse_width = int(
                config.min_pulse_us
                + angle_ratio * (config.max_pulse_us - config.min_pulse_us)
            )

            # Clamp to valid pulse range
            pulse_width = max(config.min_pulse_us, min(config.max_pulse_us, pulse_width))
            pulses[name] = pulse_width

        return pulses

    def get_serial_command(self) -> str:
        """
        Generate a serial command string for the Arduino firmware.

        Format: "JOINT_NAME:ANGLE,JOINT_NAME:ANGLE,...\\n"
        Each angle is rounded to integer degrees.

        Returns:
            Serial command string ready to send.
        """
        parts = []
        for name in sorted(self._servos):
            angle = self._get_safe_output_angle(self._servos[name])
            parts.append(f"{name}:{int(angle)}")
        return ",".join(parts) + "\n"

    def get_state_summary(self) -> str:
        """Return a human-readable summary of all servo states."""
        age = self.seconds_since_last_output()
        age_text = "never" if math.isinf(age) else f"{age:.1f}s ago"
        lines = [f"  last confirmed delivery: {age_text}"]
        for name, servo in sorted(self._servos.items()):
            lines.append(
                f"  {name:15s}  →  cur={servo.current_angle:6.1f}°  "
                f"tgt={servo.target_angle:6.1f}°  [{servo.state.value}]  "
                f"pin={servo.config.pin}  errors={servo.error_count}"
            )
        return "\n".join(lines)
