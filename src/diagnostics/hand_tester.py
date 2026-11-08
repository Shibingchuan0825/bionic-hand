#!/usr/bin/env python3
"""
Bionic Dexterous Hand - Hardware Tester / Diagnostics.

A standalone utility to test the dexterous hand without the camera /
MediaPipe pipeline. It drives the servos (real or simulated) through
defined motions so you can verify, in isolation:

  * Serial link integrity           (``ping``)
  * Servo inventory / pin mapping   (``list``)
  * Per-servo range of motion       (``sweep --servo <name>``)
  * Coordinated full-hand motion    (``sweep --all``)
  * Predefined gesture playback     (``gestures``)
  * Continuous-operation soak test  (``soak``)

Examples:
    python -m src.diagnostics.hand_tester --real ping
    python -m src.diagnostics.hand_tester --real sweep --all
    python -m src.diagnostics.hand_tester --real gestures --cycles 3
    python -m src.diagnostics.hand_tester --real soak --iterations 30

Modes:
    By default the tester runs against the SIMULATED serial backend
    (no hardware). Pass ``--real`` to talk to the physical Arduino on
    the port configured in hand_config.yaml, or ``--sim`` to force
    simulation.
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from ..motor_control.servo_manager import ServoManager
from ..motor_control.serial_controller import (
    SerialController,
    RealSerial,
    SimulatedSerial,
    is_ping_response,
    parse_ack_count,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "hand_config.yaml"

# Convergence tolerance (degrees) when deciding a servo has reached its target.
CONVERGED_EPS = 0.5

# Bounds for the per-move deadline.  The deadline is derived from the actual
# angular distance and the configured speed, so a slow 180-degree move is not
# cut off by a value tuned for a 20-degree one, and a short move does not wait
# on a fixed ten-second timer.
MIN_TRANSITION_TIME = 1.0
MAX_TRANSITION_TIME = 30.0


def _read_limits(limits: object) -> Optional[Tuple[float, float]]:
    """Return a finite, ordered ``(min, max)`` pair, or ``None`` if unusable."""
    if not isinstance(limits, dict):
        return None
    try:
        lower = float(limits["min"])
        upper = float(limits["max"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return None
    return lower, upper


# ---------------------------------------------------------------------------
# Config / logging helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the standalone tester."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------

class HandTester:
    """
    Drives the bionic hand's servos through test motions.

    Uses the same ServoManager + SerialController pipeline as the main
    application, but drives it directly (no camera / MediaPipe), so the
    mechanical hand and serial link can be verified independently.

    Unlike the buggy main control loop, this tester explicitly calls
    ``ServoManager.update_interpolation()`` each step, so the current
    angles actually move toward their targets and the serial commands
    reflect real motion.
    """

    def __init__(
        self,
        config: dict,
        simulate: bool = True,
        speed: float = 120.0,
        verbose: bool = False,
        full_range: bool = False,
        config_path: Optional[str] = None,
    ):
        self.config = config
        self.simulate = simulate
        self.full_range = full_range
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- Servo manager ---
        pin_assignments = config.get("servos", {}).get("pins", {})
        if not pin_assignments:
            raise ValueError("No 'servos.pins' found in config")
        self.manager = ServoManager(pin_assignments)

        limits = config.get("servos", {}).get("angle_limits", {})
        self.min_angle = float(limits.get("min_angle", 0.0))
        self.max_angle = float(limits.get("max_angle", 180.0))
        self.default_angle = float(limits.get("default_angle", 90.0))

        # The manifest is the authoritative source for both the bench window
        # and the nominal range.  A relative manifest path is resolved against
        # the config file that named it, not the repository root, so a config
        # kept next to its own manifest keeps working when passed with -c.
        manifest_path = config.get("hardware_manifest")
        if manifest_path:
            path = Path(manifest_path)
            if not path.is_absolute():
                # Resolve against the config file's directory first, with the
                # repository root as a fallback for an existing root-relative
                # path.  Silently resolving to a missing file would drop the
                # bench window and run the axis through its full nominal range.
                base = Path(config_path).parent if config_path else Path(__file__).parent.parent.parent
                candidate = base / path
                if not candidate.exists():
                    root_candidate = Path(__file__).parent.parent.parent / path
                    if root_candidate.exists():
                        candidate = root_candidate
                path = candidate
            try:
                with open(path, "r", encoding="utf-8") as stream:
                    manifest = yaml.safe_load(stream) or {}
            except OSError as exc:
                raise ValueError(f"Unable to load hardware manifest {path}: {exc}") from exc
            axis_specs = manifest.get("axes", [])
            if isinstance(axis_specs, dict):
                axis_specs = [
                    dict(spec, name=name) if isinstance(spec, dict) else {"name": name}
                    for name, spec in axis_specs.items()
                ]
            if not isinstance(axis_specs, list):
                raise ValueError("Hardware manifest axes must be a list or mapping")
            axis_specs = {
                axis.get("name"): axis
                for axis in axis_specs
                if isinstance(axis, dict) and axis.get("role", "active") == "active"
            }
        else:
            axis_specs = {}

        self._axis_specs = axis_specs
        self._safe_limits: Dict[str, Tuple[float, float]] = {}
        self._nominal_limits: Dict[str, Tuple[float, float]] = {}
        for name in self.manager.servo_names:
            spec = axis_specs.get(name, {})
            safe = _read_limits(spec.get("soft_limits"))
            if safe is not None:
                self._safe_limits[name] = safe
            nominal = _read_limits(spec.get("nominal_limits"))
            if nominal is not None:
                self._nominal_limits[name] = nominal

        # A real-hardware run must not silently fall back to the protocol's
        # 0-180 span just because the manifest is missing or incomplete: that
        # is exactly the range no mechanism has been cleared for yet.
        if not simulate:
            if not axis_specs:
                raise ValueError(
                    "Real-hardware mode requires a hardware_manifest with per-axis limits"
                )
            required = self._nominal_limits if full_range else self._safe_limits
            missing = [n for n in self.manager.servo_names if n not in required]
            if missing:
                kind = "nominal_limits" if full_range else "soft_limits"
                raise ValueError(
                    f"Real-hardware mode requires {kind} for every axis; "
                    f"missing or invalid for: {', '.join(sorted(missing))}"
                )

        # Apply the active window and test speed to every servo config.
        for name, servo in self.manager._servos.items():
            lo, hi = self._active_limits(name)
            servo.config.min_angle = lo
            servo.config.max_angle = hi
            servo.config.default_angle = self._axis_default(name)
            servo.current_angle = servo.config.default_angle
            servo.target_angle = servo.config.default_angle
            servo.last_command_time = time.monotonic()
            servo.config.speed_limit = speed

        # --- Serial backend ---
        if simulate:
            self.serial = SerialController(SimulatedSerial(verbose=verbose))
            self.logger.info("SIMULATION mode (no hardware)")
        else:
            serial_cfg = config.get("serial", {})
            self.serial = SerialController(
                RealSerial(
                    port=serial_cfg.get("port", "COM3"),
                    baudrate=serial_cfg.get("baudrate", 115200),
                    timeout=serial_cfg.get("timeout", 0.1),
                )
            )
            self.logger.info("HARDWARE mode on %s", serial_cfg.get("port", "COM3"))

        # Send statistics (for soak test)
        self._send_count = 0
        self._send_errors = 0
        self._ack_errors = 0
        self._sequence = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open the serial connection (real or simulated)."""
        if self.serial.connect():
            self.logger.info(
                "[OK] Connected (%s)", "simulated" if self.simulate else "real serial"
            )
            return True
        self.logger.error("[FAIL] Failed to connect")
        return False

    def safe_shutdown(self) -> None:
        """Return servos to default pose and close the connection."""
        try:
            # Continue until the interpolated state reaches the configured
            # neutral pose; a fixed number of packets is not enough after a
            # full-range test.
            self.manager.set_target_angles(self._default_targets())
            deadline = time.monotonic() + self._transition_budget(self._default_targets())
            reached_default = False
            while time.monotonic() < deadline:
                self.manager.update_interpolation(0.02)
                if not self.serial.is_connected:
                    self.logger.warning("Serial disconnected during shutdown")
                    break
                if not self._send_current_angles(require_ack=True):
                    self.logger.warning("Could not confirm default pose during shutdown")
                    break
                current = self.manager.get_all_current_angles()
                target = self.manager.get_all_target_angles()
                reached_default = all(
                    abs(current[name] - target[name]) < CONVERGED_EPS
                    for name in target
                )
                if reached_default:
                    break
                time.sleep(0.02)
            if not reached_default:
                self.logger.warning("Default pose was not reached before shutdown")
        finally:
            self.serial.disconnect()
            self.logger.info("Disconnected")

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _axis_default(self, name: str) -> float:
        """Return a manifest neutral angle, falling back to the app default."""
        try:
            return float(self._axis_specs.get(name, {}).get("neutral_angle", self.default_angle))
        except (TypeError, ValueError):
            return self.default_angle

    def _default_targets(self) -> Dict[str, float]:
        """Return the configured neutral pose for all known servos."""
        return {name: self._axis_default(name) for name in self.manager.servo_names}

    def _axis_limits(self, name: str) -> Tuple[float, float]:
        """Return the active test limits for one axis."""
        return self._active_limits(name)

    def _active_limits(self, name: str) -> Tuple[float, float]:
        """Return the window an axis may move in for this run.

        The manifest bench window is the default.  ``--full-range`` selects the
        manifest's nominal limits, which are still the protocol's declared
        range rather than a measured mechanical limit; the config-wide values
        are only a fallback for axes the manifest does not describe.
        """
        if self.full_range:
            if name in self._nominal_limits:
                return self._nominal_limits[name]
        elif name in self._safe_limits:
            return self._safe_limits[name]
        return self.min_angle, self.max_angle

    def _transition_budget(self, targets: Dict[str, float]) -> float:
        """Return the time budget for a move, derived from distance and speed."""
        current = self.manager.get_all_current_angles()
        largest_delta = 0.0
        for name, target in targets.items():
            if name not in current:
                continue
            delta = abs(float(target) - current[name])
            if math.isfinite(delta):
                largest_delta = max(largest_delta, delta)

        speeds = [
            servo.config.speed_limit
            for servo in self.manager._servos.values()
            if servo.config.speed_limit > 0
        ]
        slowest_speed = min(speeds) if speeds else 0.0
        if slowest_speed <= 0:
            estimate = MAX_TRANSITION_TIME
        else:
            estimate = largest_delta / slowest_speed
        # Allow for a slow serial link and interpolation granularity on top of
        # the ideal travel time.
        estimate = estimate * 2.0 + 1.0
        return max(MIN_TRANSITION_TIME, min(MAX_TRANSITION_TIME, estimate))

    def _expected_ack_count(self, command: str) -> int:
        """Count named fields in a serialized command."""
        return sum(1 for field in command.strip().split(",") if ":" in field)

    def _next_sequence(self) -> int:
        """Return the next protocol sequence id, wrapping within the wire range."""
        self._sequence = (self._sequence + 1) % 1000
        return self._sequence

    def _send_current_angles(self, *, require_ack: bool = False) -> bool:
        """Serialize current servo angles and send to the hardware."""
        command = self.manager.get_serial_command()
        self._send_count += 1
        if require_ack:
            # Carry a sequence id so a delayed acknowledgment from an earlier
            # frame cannot be mistaken for this one's.
            response = self.serial.send_and_receive(
                command, timeout=0.5, seq=self._next_sequence()
            )
            ok = parse_ack_count(response) == self._expected_ack_count(command)
            if not ok:
                self._ack_errors += 1
                self.logger.error(
                    "Invalid or partial ACK for %d fields: %r",
                    self._expected_ack_count(command), response,
                )
        else:
            ok = self.serial.send_command(command)
        if not ok:
            self._send_errors += 1
        return ok

    def _drive_to(
        self,
        targets: Dict[str, float],
        hold: float = 0.5,
        step: float = 0.02,
    ) -> bool:
        """
        Animate all servos from their current position to ``targets``.

        This is the correct control pipeline: set targets, interpolate the
        current angles toward them over time, and stream the result out.  Every
        diagnostic packet requires a complete firmware acknowledgment so a
        missing or partial command stops the motion test.
        """
        self.manager.set_target_angles(targets)
        # Say so when the requested pose was clamped, otherwise a bench log
        # showing "index_mcp: 30" looks like the axis actually went there.
        effective = self.manager.get_all_target_angles()
        clamped = {
            name: effective[name]
            for name, value in targets.items()
            if name in effective and abs(effective[name] - float(value)) > 1e-6
        }
        if clamped:
            self.logger.info(
                "Targets clamped to the active window: %s", clamped
            )

        # A monotonic clock, not wall time: a system clock adjustment during a
        # long sweep must not extend or cancel the deadline.
        budget = self._transition_budget(targets)
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            self.manager.update_interpolation(step)
            if not self._send_current_angles(require_ack=True):
                self.logger.error("Stopping motion after missing or partial ACK")
                return False

            current = self.manager.get_all_current_angles()
            target = self.manager.get_all_target_angles()
            if all(abs(current[k] - target[k]) < CONVERGED_EPS for k in target):
                if hold > 0:
                    time.sleep(hold)
                return True
            time.sleep(step)

        self.logger.error("Motion did not converge within %.1f s", budget)
        return False

    @staticmethod
    def _build_sweep_sequence(
        default: float, lo: float, hi: float, step_deg: float
    ) -> List[float]:
        """Generate default → lo → hi → default with ``step_deg`` resolution."""
        seq: List[float] = []
        a = default
        while a > lo:
            a = max(lo, a - step_deg)
            seq.append(a)
        a = lo
        while a < hi:
            a = min(hi, a + step_deg)
            seq.append(a)
        seq.append(default)
        return seq

    def _load_gestures(self) -> List[Tuple[str, Dict[str, float]]]:
        """Return validated ``[(name, angle_dict), ...]`` gestures.

        A gesture that does not name every active axis is rejected rather than
        padded with the neutral angle: silently neutral-filling makes an
        incomplete gesture look like a successful hand pose, and the extra axes
        it moved would leave the rig in a state nobody specified.
        """
        gestures = self.config.get("gestures", {})
        if not gestures:
            raise ValueError("No 'gestures' section found in config")

        active_names = set(self.manager.servo_names)
        result: List[Tuple[str, Dict[str, float]]] = []
        for name, gesture in gestures.items():
            angles = gesture.get("angles", {}) if isinstance(gesture, dict) else {}
            if not isinstance(angles, dict) or not angles:
                raise ValueError(f"Gesture '{name}' has no angles")
            unknown = sorted(set(angles) - active_names)
            if unknown:
                raise ValueError(f"Gesture '{name}' names unknown axes: {', '.join(unknown)}")
            missing = sorted(active_names - set(angles))
            if missing:
                raise ValueError(
                    f"Gesture '{name}' is missing active axes: {', '.join(missing)}"
                )
            result.append((name, {axis: float(value) for axis, value in angles.items()}))
        return result

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Send PING and require the Arduino's canonical PONG response."""
        self.logger.info("Pinging hardware...")
        start = time.perf_counter()
        response = self.serial.send_and_receive("PING", timeout=0.5)
        latency_ms = (time.perf_counter() - start) * 1000

        if is_ping_response(response):
            self.logger.info("PONG received in %.0f ms", latency_ms)
            return True
        self.logger.warning("Unexpected PING response after %.0f ms: %r", latency_ms, response)
        return False

    def list_servos(self) -> None:
        """Print every configured servo with channel, default, and window.

        The manifest's PCA9685 channel is the authoritative output, not the
        legacy ``servos.pins`` value, which is only an inventory label while
        PCA9685 mode is active.
        """
        print(f"{'servo':15s}  {'chan':>4s}  {'pin':>4s}   default    window")
        print("-" * 56)
        for name, servo in sorted(self.manager._servos.items()):
            channel = self._axis_specs.get(name, {}).get("channel")
            channel_text = f"{channel:>4}" if isinstance(channel, int) else "   -"
            print(
                f"{name:15s}  {channel_text}  {servo.config.pin:>4d}   "
                f"{servo.config.default_angle:5.1f}°    "
                f"[{servo.config.min_angle:.0f}, {servo.config.max_angle:.0f}]°"
            )

    def print_status(self) -> None:
        """Print current servo state summary."""
        print("\n" + self.manager.get_state_summary() + "\n")

    def sweep_servo(self, name: str, hold: float = 0.3, step_deg: float = 15.0) -> bool:
        """
        Sweep a single servo through its full range and back to default.

        Good for verifying a single joint's wiring, direction, and travel.
        Returns ``False`` as soon as a command is not fully acknowledged.
        """
        if name not in self.manager.servo_names:
            raise KeyError(f"Unknown servo: {name} (available: {self.manager.servo_names})")

        lo, hi = self._axis_limits(name)
        seq = self._build_sweep_sequence(
            self._axis_default(name), lo, hi, step_deg
        )
        self.logger.info("Sweeping '%s' through %d positions", name, len(seq))
        for pos in seq:
            self.logger.info("  %s -> %.0f°", name, pos)
            if not self._drive_to({name: pos}, hold=hold):
                return False
        return True

    def sweep_all(self, hold: float = 0.5) -> bool:
        """Sweep every servo together: default → min → max → default."""
        names = self.manager.servo_names
        neutral = self._default_targets()
        positions = [
            ("default", neutral),
            ("min", {n: self._axis_limits(n)[0] for n in names}),
            ("max", {n: self._axis_limits(n)[1] for n in names}),
            ("default", neutral),
        ]
        for label, targets in positions:
            self.logger.info("All servos -> %s", label)
            if not self._drive_to(targets, hold=hold):
                return False
        return True

    def play_gestures(self, cycles: int = 1, hold: float = 1.0) -> bool:
        """Play gestures in sequence, returning false on an invalid ACK."""
        if cycles < 1:
            raise ValueError("cycles must be at least 1")
        gestures = self._load_gestures()

        for cycle in range(cycles):
            self.logger.info("--- Gesture cycle %d/%d ---", cycle + 1, cycles)
            for name, targets in gestures:
                self.logger.info("  %s: %s", name, targets)
                if not self._drive_to(targets, hold=hold):
                    return False

            # Return to relaxed open hand
            self.logger.info("  -> default pose")
            if not self._drive_to(self._default_targets(), hold=0.5):
                return False
        return True

    def soak(self, iterations: int = 10) -> bool:
        """
        Stress test: rapidly cycle gestures for ``iterations`` rounds and
        report the serial send error rate. Verifies continuous operation.
        """
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        gestures = self._load_gestures()
        self._send_count = 0
        self._send_errors = 0
        self._ack_errors = 0

        self.logger.info("Soak test: %d iterations x %d gestures", iterations, len(gestures))
        start = time.perf_counter()

        for i in range(iterations):
            for name, targets in gestures:
                if not self._drive_to(targets, hold=0.05, step=0.01):
                    self.logger.error(
                        "Soak test stopped at iteration %d/%d, gesture '%s'",
                        i + 1,
                        iterations,
                        name,
                    )
                    return False

        elapsed = time.perf_counter() - start
        error_rate = 100.0 * self._send_errors / max(1, self._send_count)
        self.logger.info(
            "Soak complete: %d commands in %.1fs (%.0f cmds/s)",
            self._send_count, elapsed, self._send_count / max(elapsed, 1e-9),
        )
        self.logger.info(
            "Send errors: %d / %d (%.2f%%)",
            self._send_errors, self._send_count, error_rate,
        )
        self.logger.info(
            "ACK errors: %d", self._ack_errors,
        )
        return self._send_errors == 0 and self._ack_errors == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bionic Dexterous Hand - Hardware Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--real", action="store_true", help="Use real serial hardware")
    parser.add_argument("--sim", action="store_true", help="Force simulation mode")
    parser.add_argument("--speed", type=float, default=120.0,
                        help="Servo speed limit in °/s (default 120)")
    parser.add_argument("--verbose", action="store_true", help="Print every serial command")
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Use the manifest nominal limits (the protocol 0--180 span) instead "
             "of the manifest bench-test window; still not a measured mechanical limit",
    )
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ping", help="Test the serial link with PING/PONG")
    sub.add_parser("list", help="List servos and pin assignments")
    sub.add_parser("status", help="Print current servo state")

    p_sweep = sub.add_parser("sweep", help="Sweep servos through their range")
    p_sweep.add_argument("--servo", type=str, help="Sweep a single servo by name")
    p_sweep.add_argument("--all", action="store_true", help="Sweep all servos together")

    p_gestures = sub.add_parser("gestures", help="Play configured gestures")
    p_gestures.add_argument("--cycles", type=int, default=1)

    p_soak = sub.add_parser("soak", help="Stress test (rapid gesture cycling)")
    p_soak.add_argument("--iterations", type=int, default=10)

    args = parser.parse_args()

    if args.speed <= 0 or not math.isfinite(args.speed):
        parser.error("--speed must be a finite value greater than zero")
    if args.command == "gestures" and args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if args.command == "soak" and args.iterations < 1:
        parser.error("--iterations must be at least 1")

    config = load_config(args.config)
    simulate = not args.real or args.sim

    log_cfg = config.get("logging", {})
    setup_logging(args.log_level or log_cfg.get("level", "INFO"), log_cfg.get("file"))

    tester = HandTester(
        config,
        simulate=simulate,
        speed=args.speed,
        verbose=args.verbose,
        full_range=args.full_range,
        config_path=args.config,
    )

    # `list` is static and needs no connection.
    if args.command == "list":
        tester.list_servos()
        return 0

    if not tester.connect():
        return 1

    rc = 0
    try:
        if args.command == "ping":
            rc = 0 if tester.ping() else 1
        elif args.command == "status":
            tester.print_status()
        elif args.command == "sweep":
            if args.all:
                ok = tester.sweep_all()
            elif args.servo:
                ok = tester.sweep_servo(args.servo)
            else:
                parser.error("sweep requires --servo NAME or --all")
            rc = 0 if ok else 1
        elif args.command == "gestures":
            rc = 0 if tester.play_gestures(cycles=args.cycles) else 1
        elif args.command == "soak":
            rc = 0 if tester.soak(iterations=args.iterations) else 1
    except KeyboardInterrupt:
        print("\nInterrupted — returning to default pose.")
        rc = 130
    finally:
        tester.safe_shutdown()

    return rc


if __name__ == "__main__":
    sys.exit(main())
