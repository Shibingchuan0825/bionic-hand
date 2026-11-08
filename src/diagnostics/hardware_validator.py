"""Validate the software, firmware, and planned hardware channel map.

The manifest is intentionally separate from ``hand_config.yaml``: the latter
contains application settings, while this module verifies that the names and
PCA9685 channels used by each layer agree.  A planned axis is not treated as a
built physical DoF until ``physical_dof_claimed`` and an actuator mapping are
both recorded.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


DEFAULT_MANIFEST_PATH = Path(__file__).parent.parent.parent / "config" / "hardware_manifest.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "hand_config.yaml"
DEFAULT_FIRMWARE_PATH = (
    Path(__file__).parent.parent.parent / "arduino" / "bionic_hand" / "bionic_hand.ino"
)

# The serial protocol and the firmware both constrain every angle to this range.
PROTOCOL_MIN_ANGLE = 0.0
PROTOCOL_MAX_ANGLE = 180.0

# Firmware DRIVER_MODE value mapped to the manifest's controller.driver string.
_DRIVER_MODE_NAMES = {0: "direct_pwm", 1: "pca9685"}


class HardwareManifestError(ValueError):
    """Raised when a hardware manifest cannot be interpreted."""


def load_yaml(path: Path | str) -> dict:
    """Load a YAML mapping from ``path``."""
    with open(path, "r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise HardwareManifestError(f"Expected a YAML mapping: {path}")
    return value


def _axis_list(manifest: Mapping[str, Any]) -> List[dict]:
    """Return manifest axes as dictionaries with a guaranteed ``name`` key."""
    raw_axes = manifest.get("axes")
    if isinstance(raw_axes, list):
        axes = raw_axes
    elif isinstance(raw_axes, dict):
        axes = []
        for name, raw_spec in raw_axes.items():
            spec = dict(raw_spec) if isinstance(raw_spec, dict) else {}
            spec.setdefault("name", name)
            axes.append(spec)
    else:
        return []

    result: List[dict] = []
    for raw_axis in axes:
        result.append(dict(raw_axis) if isinstance(raw_axis, dict) else {})
    return result


def active_axis_specs(manifest: Mapping[str, Any]) -> Dict[str, dict]:
    """Return active axis specifications keyed by logical joint name."""
    return {
        axis["name"]: axis
        for axis in _axis_list(manifest)
        if axis.get("role", "active") == "active" and axis.get("name")
    }


_FIRMWARE_CONSTANT_PATTERNS = {
    "driver_mode": r"^\s*#define\s+DRIVER_MODE\s+(\d+)\s*$",
    "serial_baud": r"^\s*#define\s+SERIAL_BAUD\s+(\d+)\s*$",
    "servo_freq": r"^\s*#define\s+SERVO_FREQ\s+(\d+)\s*$",
    "pca9685_addr": r"^\s*#define\s+PCA9685_ADDR\s+(0[xX][0-9a-fA-F]+|\d+)\s*$",
    "num_servos": r"^\s*#define\s+NUM_SERVOS\s+(\d+)\s*$",
}


def parse_firmware_constants(source: str) -> Dict[str, int]:
    """Extract the firmware's protocol-relevant constants.

    Only constants the firmware actually declares are returned.  A missing key
    means the comparison is skipped rather than guessed, so a renamed or
    removed define shows up as a gap in coverage instead of a false pass.
    """
    constants: Dict[str, int] = {}
    for key, pattern in _FIRMWARE_CONSTANT_PATTERNS.items():
        match = re.search(pattern, source, flags=re.MULTILINE)
        if not match:
            continue
        text = match.group(1)
        constants[key] = int(text, 16) if text[:2].lower() == "0x" else int(text)
    return constants


def _as_int(value: Any) -> Optional[int]:
    """Return ``value`` as an int, accepting decimal and ``0x`` strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 16) if value[:2].lower() == "0x" else int(value)
        except ValueError:
            return None
    return None


def parse_firmware_arrays(source: str) -> Tuple[List[str], List[int]]:
    """Extract and validate active firmware name and channel arrays.

    Both driver modes are supported.  ``DRIVER_MODE 1`` (PCA9685) maps to the
    ``SERVO_CHANNELS`` array; ``DRIVER_MODE 0`` (direct PWM) maps to
    ``SERVO_PINS``, where a negative sentinel means "this axis has no pin".
    Validate only the array belonging to the mode the source actually selects,
    so the inactive branch can never be silently validated.

    The declared ``NUM_SERVOS`` count must agree with both parsed arrays.
    """
    mode_match = re.search(r"^\s*#define\s+DRIVER_MODE\s+(\d+)\s*$", source, flags=re.MULTILINE)
    mode = int(mode_match.group(1)) if mode_match else 1
    if mode not in (0, 1):
        raise HardwareManifestError(f"Firmware DRIVER_MODE={mode} is not a supported mode (0 or 1)")

    count_match = re.search(r"^\s*#define\s+NUM_SERVOS\s+(\d+)\s*$", source, flags=re.MULTILINE)
    declared_count = int(count_match.group(1)) if count_match else None

    names_match = re.search(
        r"const\s+char\s*\*\s*SERVO_NAMES\s*\[\s*NUM_SERVOS\s*\]\s*=\s*\{(.*?)\};",
        source,
        flags=re.DOTALL,
    )
    if not names_match:
        raise HardwareManifestError("SERVO_NAMES array not found in firmware")
    names = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', names_match.group(1))

    if mode == 1:
        array_label = "SERVO_CHANNELS"
        array_pattern = r"const\s+uint8_t\s+SERVO_CHANNELS\s*\[\s*NUM_SERVOS\s*\]\s*=\s*\{(.*?)\};"
    else:
        array_label = "SERVO_PINS"
        array_pattern = r"const\s+int16_t\s+SERVO_PINS\s*\[\s*NUM_SERVOS\s*\]\s*=\s*\{(.*?)\};"

    array_match = re.search(array_pattern, source, flags=re.DOTALL)
    if not array_match:
        raise HardwareManifestError(f"{array_label} array not found in firmware")
    tokens = re.findall(r"-?\d+", array_match.group(1))
    try:
        channels = [int(token) for token in tokens]
    except ValueError as exc:  # pragma: no cover - regex only captures digits
        raise HardwareManifestError(f"Invalid {array_label} value") from exc

    if declared_count is not None:
        if len(names) != declared_count:
            raise HardwareManifestError(
                f"SERVO_NAMES contains {len(names)} entries, expected NUM_SERVOS={declared_count}"
            )
        if len(channels) != declared_count:
            raise HardwareManifestError(
                f"{array_label} contains {len(channels)} entries, expected NUM_SERVOS={declared_count}"
            )
    if len(names) != len(channels):
        raise HardwareManifestError(
            f"SERVO_NAMES/{array_label} length mismatch: {len(names)} != {len(channels)}"
        )
    return names, channels


def _compare_names(
    label: str,
    expected: Iterable[str],
    actual: Iterable[str],
    *,
    require_order: bool = False,
) -> List[str]:
    """Compare names, optionally enforcing order for positional arrays."""
    expected_list = list(expected)
    actual_list = list(actual)
    issues: List[str] = []
    expected_set = set(expected_list)
    actual_set = set(actual_list)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing:
        issues.append(f"{label} missing names: {', '.join(missing)}")
    if unexpected:
        issues.append(f"{label} has unknown names: {', '.join(unexpected)}")
    if require_order and not missing and not unexpected and expected_list != actual_list:
        issues.append(f"{label} order differs: expected {expected_list}, got {actual_list}")
    return issues


def _limits_are_valid(axis: Mapping[str, Any], key: str, name: str) -> List[str]:
    limits = axis.get(key)
    if not isinstance(limits, Mapping):
        return [f"axis {name}: {key} must contain min and max"]
    if "min" not in limits or "max" not in limits:
        return [f"axis {name}: {key} must contain min and max"]
    try:
        lower = float(limits["min"])
        upper = float(limits["max"])
        neutral = float(axis.get("neutral_angle"))
    except (TypeError, ValueError):
        return [f"axis {name}: {key} and neutral_angle must be numeric"]
    # YAML happily parses `.inf` and `.nan`, and a NaN limit would pass every
    # subsequent comparison silently.
    for label, value in (("min", lower), ("max", upper), ("neutral_angle", neutral)):
        if not math.isfinite(value):
            return [f"axis {name}: {key}.{label} must be finite (got {value})"]
    if lower > upper:
        return [f"axis {name}: {key}.min is greater than max"]
    # The wire protocol and the firmware clamp to 0-180, so a limit outside
    # that range cannot be enforced and must not be recorded as if it could.
    if lower < PROTOCOL_MIN_ANGLE or upper > PROTOCOL_MAX_ANGLE:
        return [
            f"axis {name}: {key} [{lower:g}, {upper:g}] is outside the protocol "
            f"range {PROTOCOL_MIN_ANGLE:g}-{PROTOCOL_MAX_ANGLE:g}"
        ]
    if not lower <= neutral <= upper:
        return [f"axis {name}: neutral_angle is outside {key}"]
    return []


def _compare_firmware_constants(
    firmware_config: Mapping[str, Any],
    controller: Mapping[str, Any],
    hand_config: Optional[Mapping[str, Any]],
    active_axis_count: int,
) -> List[str]:
    """Compare the manifest's controller block with the firmware's defines."""
    issues: List[str] = []

    driver_mode = _as_int(firmware_config.get("driver_mode"))
    if driver_mode is not None:
        expected_driver = str(controller.get("driver", "")).strip().lower()
        firmware_driver = _DRIVER_MODE_NAMES.get(driver_mode)
        if firmware_driver is None:
            issues.append(
                f"firmware DRIVER_MODE={driver_mode} is not a supported mode (0 or 1)"
            )
        elif expected_driver and expected_driver != firmware_driver:
            issues.append(
                f"controller.driver is '{expected_driver}' but firmware DRIVER_MODE="
                f"{driver_mode} selects '{firmware_driver}'"
            )

    # The I2C address and PWM frequency only exist in PCA9685 mode.  In
    # direct-PWM mode the firmware still defines them (the constant is shared),
    # so comparing them would report a difference that cannot affect behaviour.
    if driver_mode != 0:
        firmware_address = _as_int(firmware_config.get("pca9685_addr"))
        manifest_address = _as_int(controller.get("i2c_address"))
        if firmware_address is not None and manifest_address is not None:
            if firmware_address != manifest_address:
                issues.append(
                    f"PCA9685 address differs: manifest {manifest_address:#04x}, "
                    f"firmware {firmware_address:#04x}"
                )

        firmware_frequency = _as_int(firmware_config.get("servo_freq"))
        manifest_frequency = _as_int(controller.get("pwm_frequency_hz"))
        if firmware_frequency is not None and manifest_frequency is not None:
            if firmware_frequency != manifest_frequency:
                issues.append(
                    f"PWM frequency differs: manifest {manifest_frequency} Hz, "
                    f"firmware {firmware_frequency} Hz"
                )

    firmware_count = _as_int(firmware_config.get("num_servos"))
    if firmware_count is not None and firmware_count != active_axis_count:
        issues.append(
            f"firmware NUM_SERVOS={firmware_count} but the manifest declares "
            f"{active_axis_count} active axes"
        )

    firmware_baud = _as_int(firmware_config.get("serial_baud"))
    if firmware_baud is not None and hand_config is not None:
        serial_config = hand_config.get("serial", {})
        if isinstance(serial_config, Mapping):
            configured_baud = _as_int(serial_config.get("baudrate"))
            if configured_baud is not None and configured_baud != firmware_baud:
                issues.append(
                    f"serial baudrate differs: hand_config {configured_baud}, "
                    f"firmware {firmware_baud}"
                )

    return issues


def validate_configuration(
    manifest: Mapping[str, Any],
    hand_config: Optional[Mapping[str, Any]] = None,
    *,
    joint_names: Optional[Iterable[str]] = None,
    firmware_names: Optional[Sequence[str]] = None,
    firmware_channels: Optional[Sequence[int]] = None,
    firmware_config: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return validation errors for the complete name/channel configuration."""
    issues: List[str] = []
    axes = _axis_list(manifest)
    if not axes:
        return ["manifest has no axes"]

    names = [axis.get("name") for axis in axes]
    if any(not isinstance(name, str) or not name for name in names):
        issues.append("every manifest axis must have a non-empty name")
    if len(set(names)) != len(names):
        issues.append("manifest contains duplicate axis names")

    capacity = manifest.get("controller", {}).get("channel_capacity", 16)
    try:
        capacity = int(capacity)
    except (TypeError, ValueError):
        issues.append("controller.channel_capacity must be an integer")
        capacity = 16

    channels: List[int] = []
    for axis in axes:
        name = axis.get("name", "<unnamed>")
        channel = axis.get("channel")
        if not isinstance(channel, int) or isinstance(channel, bool):
            issues.append(f"axis {name}: channel must be an integer")
            continue
        channels.append(channel)
        if channel < 0 or channel >= capacity:
            issues.append(f"axis {name}: channel {channel} outside 0..{capacity - 1}")
        if axis.get("role", "active") == "active":
            issues.extend(_limits_are_valid(axis, "soft_limits", name))
            issues.extend(_limits_are_valid(axis, "nominal_limits", name))
            if "neutral_angle" not in axis:
                issues.append(f"axis {name}: neutral_angle is required")

        claimed = bool(axis.get("physical_dof_claimed", False))
        if axis.get("physical_dof_status") in {"built", "validated"}:
            claimed = True
        if claimed and not axis.get("actuator"):
            issues.append(f"axis {name}: claimed physical DoF has no actuator mapping")

    if len(channels) != len(set(channels)):
        issues.append("manifest contains duplicate PCA/controller channels")

    active_axes = [axis for axis in axes if axis.get("role", "active") == "active"]
    active_names = [axis.get("name") for axis in active_axes]
    if len(active_axes) > capacity:
        issues.append(f"{len(active_axes)} active axes exceed {capacity} controller channels")

    active_by_channel = sorted(active_axes, key=lambda axis: axis.get("channel", -1))
    expected_firmware_names = [axis.get("name") for axis in active_by_channel]
    expected_firmware_channels = [axis.get("channel") for axis in active_by_channel]

    if hand_config is not None:
        servo_config = hand_config.get("servos", {})
        pin_assignments = servo_config.get("pins", {}) if isinstance(servo_config, Mapping) else {}
        if not isinstance(pin_assignments, Mapping):
            issues.append("hand_config.servos.pins must be a mapping")
            pin_assignments = {}
        issues.extend(_compare_names("hand_config.servos.pins", active_names, pin_assignments.keys()))

        gestures = hand_config.get("gestures", {})
        if isinstance(gestures, Mapping):
            for gesture_name, gesture in gestures.items():
                # Non-gesture keys at the same level (switches, tuning knobs)
                # are not poses and must not be validated as if they were.
                if not isinstance(gesture, Mapping):
                    continue
                if "angles" not in gesture:
                    continue
                angle_map = gesture.get("angles", {})
                if not isinstance(angle_map, Mapping):
                    issues.append(f"gesture {gesture_name}: angles must be a mapping")
                    continue
                issues.extend(_compare_names(f"gesture {gesture_name}", active_names, angle_map.keys()))

    if joint_names is not None:
        issues.extend(_compare_names("JointAngles.to_dict()", active_names, joint_names))

    if firmware_names is not None:
        firmware_name_list = list(firmware_names)
        if firmware_name_list != expected_firmware_names:
            issues.append(
                "firmware SERVO_NAMES order/channel map differs: "
                f"expected {expected_firmware_names}, got {firmware_name_list}"
            )

    if firmware_channels is not None:
        firmware_channel_list = list(firmware_channels)
        if firmware_channel_list != expected_firmware_channels:
            issues.append(
                "firmware SERVO_CHANNELS differs: "
                f"expected {expected_firmware_channels}, got {firmware_channel_list}"
            )

    if firmware_config is not None:
        controller_block = manifest.get("controller", {})
        if not isinstance(controller_block, Mapping):
            issues.append("manifest controller must be a mapping")
            controller_block = {}
        issues.extend(
            _compare_firmware_constants(
                firmware_config,
                controller_block,
                hand_config,
                len(active_axes),
            )
        )

    return issues


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate checked-in configuration from the command line."""
    parser = argparse.ArgumentParser(description="Validate hand hardware/channel configuration")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--firmware", type=Path, default=DEFAULT_FIRMWARE_PATH)
    args = parser.parse_args(argv)

    try:
        manifest = load_yaml(args.manifest)
        hand_config = load_yaml(args.config)
        firmware_source = args.firmware.read_text(encoding="utf-8")
        firmware_names, firmware_channels = parse_firmware_arrays(firmware_source)
        from ..hand_mapper.joint_mapper import JointAngles

        joint_names = list(JointAngles().to_dict().keys())
        issues = validate_configuration(
            manifest,
            hand_config,
            joint_names=joint_names,
            firmware_names=firmware_names,
            firmware_channels=firmware_channels,
            firmware_config=parse_firmware_constants(firmware_source),
        )
    except (OSError, HardwareManifestError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    if issues:
        for issue in issues:
            print(f"[FAIL] {issue}")
        return 1

    print("[OK] Hardware manifest, Python config, JointAngles, and firmware agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
