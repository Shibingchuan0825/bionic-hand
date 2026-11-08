"""
Tests for the ServoManager module.
"""

import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.motor_control.servo_manager import ServoManager, ServoConfig, ServoState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pin_assignments():
    return {
        "thumb_cmc": 2, "thumb_mcp": 3, "thumb_ip": 4,
        "index_mcp": 5, "index_pip": 6,
        "middle_mcp": 7, "middle_pip": 8,
        "ring_mcp": 9, "ring_pip": 10,
        "pinky_mcp": 11, "pinky_pip": 12,
        "wrist_rot": 13,
    }


@pytest.fixture
def servo_manager(pin_assignments):
    return ServoManager(pin_assignments)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServoManager:
    """Tests for the ServoManager class."""

    def test_initialization(self, servo_manager):
        """ServoManager should initialize with correct number of servos."""
        assert servo_manager.servo_count == 12
        assert "index_mcp" in servo_manager.servo_names
        assert "thumb_cmc" in servo_manager.servo_names

    def test_default_angles(self, servo_manager):
        """All servos should start at default angle (90°)."""
        angles = servo_manager.get_all_current_angles()
        for name, angle in angles.items():
            assert angle == 90.0, f"{name} should start at 90°, got {angle}"

    def test_set_target_angles(self, servo_manager):
        """Setting target angles should update targets."""
        targets = {
            "index_mcp": 120.0,
            "index_pip": 160.0,
            "middle_mcp": 100.0,
        }
        servo_manager.set_target_angles(targets)

        target_angles = servo_manager.get_all_target_angles()
        assert target_angles["index_mcp"] == 120.0
        assert target_angles["index_pip"] == 160.0
        assert target_angles["middle_mcp"] == 100.0

        # Untargeted servos should stay at default
        assert target_angles["thumb_cmc"] == 90.0

    def test_angle_clamping(self, servo_manager):
        """Angles outside [0, 180] should be clamped."""
        targets = {
            "index_mcp": 200.0,  # Too high → clamp to 180
            "index_pip": -50.0,   # Too low → clamp to 0
            "middle_mcp": 90.0,   # Valid → unchanged
        }
        servo_manager.set_target_angles(targets)

        target_angles = servo_manager.get_all_target_angles()
        assert target_angles["index_mcp"] == 180.0
        assert target_angles["index_pip"] == 0.0
        assert target_angles["middle_mcp"] == 90.0

    def test_set_single_angle(self, servo_manager):
        """Setting a single servo should only affect that servo."""
        servo_manager.set_single_angle("thumb_mcp", 45.0)

        targets = servo_manager.get_all_target_angles()
        assert targets["thumb_mcp"] == 45.0
        assert targets["thumb_cmc"] == 90.0  # Unchanged

    def test_interpolation(self, servo_manager):
        """Interpolation should move current angles toward targets."""
        # Set target far from current
        servo_manager.set_single_angle("index_mcp", 180.0)

        # Initial current should be default
        assert servo_manager.get_current_angle("index_mcp") == 90.0

        # Run one interpolation step (large dt to allow full movement)
        servo_manager.update_interpolation(dt=10.0)

        # Should now be at target (no speed limit by default)
        assert servo_manager.get_current_angle("index_mcp") == 180.0

    def test_interpolation_speed_limit(self, servo_manager):
        """Speed-limited servo should move gradually."""
        # Set a speed limit
        servo_manager._servos["index_mcp"].config.speed_limit = 30.0  # 30°/s

        # Set target far from current
        servo_manager.set_single_angle("index_mcp", 180.0)

        # Small time step
        servo_manager.update_interpolation(dt=1.0)

        # Should have moved at most 30° in 1 second
        angle = servo_manager.get_current_angle("index_mcp")
        assert 119.0 <= angle <= 121.0, f"Expected ~120°, got {angle}"

        # Another step
        servo_manager.update_interpolation(dt=1.0)
        angle = servo_manager.get_current_angle("index_mcp")
        assert 149.0 <= angle <= 151.0, f"Expected ~150°, got {angle}"

    def test_default_pose(self, servo_manager):
        """set_default_pose should reset all targets to 90°."""
        # Move away from default
        servo_manager.set_single_angle("index_mcp", 30.0)
        servo_manager.set_single_angle("thumb_mcp", 150.0)

        # Reset
        servo_manager.set_default_pose()

        targets = servo_manager.get_all_target_angles()
        for name, angle in targets.items():
            assert angle == 90.0, f"{name} should be 90° after reset, got {angle}"

    def test_request_safe_pose_resets_targets(self, servo_manager):
        """A safe-pose request resets targets; it is not a physical e-stop."""
        servo_manager.set_single_angle("index_mcp", 180.0)
        servo_manager.request_safe_pose("test")

        targets = servo_manager.get_all_target_angles()
        assert targets["index_mcp"] == 90.0

    def test_delivery_tracking_is_separate_from_intent(self, servo_manager):
        """Setting a target must not look like a confirmed delivery."""
        assert math.isinf(servo_manager.seconds_since_last_output())

        servo_manager.set_single_angle("index_mcp", 120.0)
        assert math.isinf(servo_manager.seconds_since_last_output())

        servo_manager.note_output_acknowledged()
        assert servo_manager.seconds_since_last_output() < 1.0

    def test_output_failure_counts_errors(self, servo_manager):
        servo_manager.note_output_failure()
        servo_manager.note_output_failure()

        summary = servo_manager.get_state_summary()
        assert "errors=2" in summary
        assert "last confirmed delivery: never" in summary

    def test_unknown_servo(self, servo_manager):
        """Setting an unknown servo should log warning but not crash."""
        servo_manager.set_single_angle("nonexistent_joint", 90.0)
        # Should not raise

    def test_get_serial_command(self, servo_manager):
        """Serial command should be properly formatted."""
        # Set specific angles
        servo_manager.set_single_angle("index_mcp", 120)
        servo_manager.set_single_angle("index_pip", 160)

        # Apply interpolation to move current angles to targets
        servo_manager.update_interpolation(dt=10.0)

        command = servo_manager.get_serial_command()
        assert command.endswith("\n")
        assert "index_mcp:120" in command
        assert "index_pip:160" in command

    def test_serial_command_uses_safe_angle_when_stale(self, servo_manager):
        """Serial output should return stale servos to their neutral angle."""
        servo_manager.set_single_angle("index_mcp", 30.0)
        servo_manager.update_interpolation(dt=10.0)
        servo = servo_manager._servos["index_mcp"]
        servo.last_command_time = time.monotonic() - servo_manager.STALE_TIMEOUT_SEC - 1.0

        command = servo_manager.get_serial_command()

        assert "index_mcp:90" in command
        assert servo.state is ServoState.DISCONNECTED

    def test_pulse_widths(self, servo_manager):
        """Pulse widths should map correctly."""
        servo_manager.set_single_angle("index_mcp", 0.0)
        servo_manager.update_interpolation(dt=10.0)

        pulses = servo_manager.get_pulse_widths()
        assert pulses["index_mcp"] == 500  # 0° → min pulse

        servo_manager.set_single_angle("index_mcp", 180.0)
        servo_manager.update_interpolation(dt=10.0)

        pulses = servo_manager.get_pulse_widths()
        assert pulses["index_mcp"] == 2500  # 180° → max pulse

        servo_manager.set_single_angle("index_mcp", 90.0)
        servo_manager.update_interpolation(dt=10.0)

        pulses = servo_manager.get_pulse_widths()
        assert pulses["index_mcp"] == 1500  # 90° → center pulse

    def test_state_summary(self, servo_manager):
        """State summary should contain servo names and states."""
        summary = servo_manager.get_state_summary()
        assert "index_mcp" in summary
        assert "ok" in summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
