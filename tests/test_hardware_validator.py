"""Tests for the authoritative hardware manifest validator."""

import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.hardware_validator import (
    HardwareManifestError,
    active_axis_specs,
    load_yaml,
    parse_firmware_arrays,
    parse_firmware_constants,
    validate_configuration,
)
from src.hand_mapper.joint_mapper import JointAngles


ROOT = Path(__file__).parent.parent


def test_checked_in_hardware_configuration_is_consistent():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    config = load_yaml(ROOT / "config" / "hand_config.yaml")
    firmware = (ROOT / "arduino" / "bionic_hand" / "bionic_hand.ino").read_text(
        encoding="utf-8"
    )
    firmware_names, firmware_channels = parse_firmware_arrays(firmware)

    issues = validate_configuration(
        manifest,
        config,
        joint_names=JointAngles().to_dict().keys(),
        firmware_names=firmware_names,
        firmware_channels=firmware_channels,
        firmware_config=parse_firmware_constants(firmware),
    )

    assert issues == []
    assert len(active_axis_specs(manifest)) == 12


def test_direct_pwm_mode_parses_the_pin_array_not_the_pca_array():
    """Direct-PWM firmware must be validated against SERVO_PINS.

    The two arrays are not interchangeable: PCA channels are 0-based output
    numbers, direct-PWM values are physical pin numbers.  Parsing the wrong one
    would let a board whose pins do not match the manifest pass validation.
    """
    source = """
#define DRIVER_MODE 0
#define NUM_SERVOS 1
const char* SERVO_NAMES[NUM_SERVOS] = {"index_mcp"};
const uint8_t SERVO_CHANNELS[NUM_SERVOS] = {0};
const int16_t SERVO_PINS[NUM_SERVOS] = {5};
"""

    names, channels = parse_firmware_arrays(source)

    assert names == ["index_mcp"]
    assert channels == [5], "must read SERVO_PINS, not SERVO_CHANNELS"


def test_direct_pwm_mode_without_a_pin_array_is_rejected():
    source = """
#define DRIVER_MODE 0
#define NUM_SERVOS 1
const char* SERVO_NAMES[NUM_SERVOS] = {"index_mcp"};
const uint8_t SERVO_CHANNELS[NUM_SERVOS] = {0};
"""

    with pytest.raises(HardwareManifestError, match="SERVO_PINS"):
        parse_firmware_arrays(source)


def test_unknown_driver_mode_is_rejected():
    source = """
#define DRIVER_MODE 2
const char* SERVO_NAMES[NUM_SERVOS] = {"index_mcp"};
"""

    with pytest.raises(HardwareManifestError, match="DRIVER_MODE"):
        parse_firmware_arrays(source)


def test_duplicate_channel_is_rejected():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    manifest["axes"][1]["channel"] = manifest["axes"][0]["channel"]

    issues = validate_configuration(manifest)

    assert any("duplicate" in issue.lower() for issue in issues)


def test_num_servos_must_match_firmware_arrays():
    source = """
#define DRIVER_MODE 1
#define NUM_SERVOS 2
const char* SERVO_NAMES[NUM_SERVOS] = {"index_mcp"};
const uint8_t SERVO_CHANNELS[NUM_SERVOS] = {0};
"""

    with pytest.raises(HardwareManifestError, match="NUM_SERVOS"):
        parse_firmware_arrays(source)


def test_claimed_physical_dof_requires_an_actuator():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    manifest["axes"][0]["physical_dof_claimed"] = True

    issues = validate_configuration(manifest)

    assert any("no actuator mapping" in issue for issue in issues)


def test_unknown_gesture_name_is_rejected():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    config = load_yaml(ROOT / "config" / "hand_config.yaml")
    config["gestures"]["open_hand"]["angles"]["not_a_joint"] = 90

    issues = validate_configuration(manifest, config)

    assert any("unknown names" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Firmware constants
# ---------------------------------------------------------------------------

def test_firmware_constants_are_parsed_including_hex_address():
    source = """
#define SERIAL_BAUD 115200
#define SERVO_FREQ    50
#define PCA9685_ADDR 0x40
#define NUM_SERVOS 12
#define DRIVER_MODE 1
"""

    constants = parse_firmware_constants(source)

    assert constants == {
        "serial_baud": 115200,
        "servo_freq": 50,
        "pca9685_addr": 0x40,
        "num_servos": 12,
        "driver_mode": 1,
    }


def test_missing_firmware_constants_are_skipped_not_guessed():
    assert parse_firmware_constants("#define DRIVER_MODE 1") == {"driver_mode": 1}


def test_controller_mismatch_is_rejected():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    config = load_yaml(ROOT / "config" / "hand_config.yaml")
    # The manifest selects direct PWM, so firmware advertising PCA9685 is a
    # mismatch — and it is the PCA9685 branch where the address and frequency
    # comparisons are meaningful.
    firmware_config = {
        "driver_mode": 1,
        "serial_baud": 9600,
        "servo_freq": 60,
        "pca9685_addr": 0x41,
        "num_servos": 13,
    }

    issues = validate_configuration(manifest, config, firmware_config=firmware_config)

    assert any("controller.driver" in issue for issue in issues)
    assert any("PCA9685 address differs" in issue for issue in issues)
    assert any("PWM frequency differs" in issue for issue in issues)
    assert any("serial baudrate differs" in issue for issue in issues)
    assert any("NUM_SERVOS=13" in issue for issue in issues)


def test_i2c_settings_are_not_compared_in_direct_pwm_mode():
    """A direct-PWM build must not be failed on PCA9685-only settings."""
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    config = load_yaml(ROOT / "config" / "hand_config.yaml")
    firmware_config = {
        "driver_mode": 0,
        "serial_baud": 115200,
        "servo_freq": 60,          # differs from the manifest
        "pca9685_addr": 0x41,      # differs from the manifest
        "num_servos": 12,
    }

    issues = validate_configuration(manifest, config, firmware_config=firmware_config)

    assert not any("PCA9685 address differs" in issue for issue in issues)
    assert not any("PWM frequency differs" in issue for issue in issues)


def test_unsupported_driver_mode_is_rejected():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")

    issues = validate_configuration(manifest, firmware_config={"driver_mode": 7})

    assert any("not a supported mode" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

def test_non_finite_limits_are_rejected():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    manifest["axes"][0]["soft_limits"] = {"min": float("inf"), "max": 100}

    issues = validate_configuration(manifest)

    assert any("must be finite" in issue for issue in issues)


def test_nan_limits_are_rejected():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    manifest["axes"][0]["nominal_limits"] = {"min": float("nan"), "max": 180}

    issues = validate_configuration(manifest)

    assert any("must be finite" in issue for issue in issues)


def test_limits_outside_the_protocol_range_are_rejected():
    """The wire protocol and firmware clamp to 0-180 degrees."""
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    manifest["axes"][0]["soft_limits"] = {"min": -20, "max": 200}
    manifest["axes"][0]["neutral_angle"] = 90

    issues = validate_configuration(manifest)

    assert any("outside the protocol range" in issue for issue in issues)


def test_yaml_inf_is_parsed_as_a_float_and_rejected():
    """YAML turns ``.inf`` into a real float, which must not pass silently."""
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    limits = yaml.safe_load("min: .inf\nmax: 100\n")
    assert math.isinf(limits["min"])
    manifest["axes"][0]["soft_limits"] = limits

    issues = validate_configuration(manifest)

    assert any("must be finite" in issue for issue in issues)


def test_non_numeric_limits_are_rejected():
    manifest = load_yaml(ROOT / "config" / "hardware_manifest.yaml")
    manifest["axes"][0]["soft_limits"] = {"min": "wide", "max": 100}

    issues = validate_configuration(manifest)

    assert any("must be numeric" in issue for issue in issues)
