"""
Tests for the HandTester diagnostic module (simulation mode only).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.diagnostics.hand_tester import HandTester


def make_config():
    return {
        "servos": {
            "pins": {"thumb_mcp": 3, "index_mcp": 5, "index_pip": 6},
            "angle_limits": {"min_angle": 0, "max_angle": 180, "default_angle": 90},
        },
        "gestures": {
            "open": {
                "description": "open",
                "angles": {"thumb_mcp": 160, "index_mcp": 160, "index_pip": 160},
            },
            "fist": {
                "description": "fist",
                "angles": {"thumb_mcp": 30, "index_mcp": 30, "index_pip": 30},
            },
        },
        "serial": {"port": "COM3", "baudrate": 115200, "timeout": 0.1},
        "simulation": {"enabled": True},
    }


@pytest.fixture
def tester():
    # Very high speed limit so tests converge in a single step.
    t = HandTester(make_config(), simulate=True, speed=1e6)
    yield t
    t.safe_shutdown()


def test_ping_simulated(tester):
    """PING should receive an echo in simulation mode."""
    assert tester.connect()
    assert tester.ping()


def test_sweep_single_servo_returns_to_default(tester):
    """A single-servo sweep should finish back at the default angle."""
    tester.connect()
    tester.sweep_servo("index_mcp", hold=0.0, step_deg=180.0)
    assert tester.manager.get_current_angle("index_mcp") == pytest.approx(90.0, abs=0.5)


def test_sweep_reaches_extremes(tester):
    """Sweeping all servos should actually drive them off the default angle."""
    tester.connect()
    tester.sweep_all(hold=0.0)

    # After the full sweep the hand should be back at default.
    for name, angle in tester.manager.get_all_current_angles().items():
        assert angle == pytest.approx(90.0, abs=0.5)


def test_play_gestures(tester):
    """Gesture playback should run and leave the hand at the default pose."""
    tester.connect()
    tester.play_gestures(cycles=1, hold=0.0)

    for name, angle in tester.manager.get_all_current_angles().items():
        assert angle == pytest.approx(90.0, abs=0.5)


def test_unknown_servo_raises(tester):
    """Sweeping a nonexistent servo should raise KeyError."""
    tester.connect()
    with pytest.raises(KeyError):
        tester.sweep_servo("nonexistent_joint")


def test_incomplete_gesture_is_rejected():
    """A gesture that omits an active axis must fail rather than be padded."""
    config = make_config()
    config["gestures"]["open"]["angles"].pop("index_pip")

    tester = HandTester(config, simulate=True, speed=1e6)
    try:
        with pytest.raises(ValueError, match="missing active axes"):
            tester._load_gestures()
    finally:
        tester.serial.disconnect()


def test_gesture_with_unknown_axis_is_rejected():
    config = make_config()
    config["gestures"]["open"]["angles"]["not_a_joint"] = 90

    tester = HandTester(config, simulate=True, speed=1e6)
    try:
        with pytest.raises(ValueError, match="unknown axes"):
            tester._load_gestures()
    finally:
        tester.serial.disconnect()


def test_zero_gesture_cycles_is_rejected(tester):
    tester.connect()
    with pytest.raises(ValueError, match="cycles"):
        tester.play_gestures(cycles=0, hold=0.0)


def test_short_move_does_not_wait_on_a_fixed_deadline(tester):
    """The deadline must scale with distance, not stay pinned at 10 s."""
    tester.connect()
    tester.manager.set_target_angles({"index_mcp": 91.0})

    assert tester._transition_budget({"index_mcp": 91.0}) < 5.0


def test_long_move_gets_a_larger_budget(tester):
    tester.connect()
    tester.manager.set_target_angles({"index_mcp": 90.0})

    short = tester._transition_budget({"index_mcp": 91.0})
    long = tester._transition_budget({"index_mcp": 180.0})

    assert long > short


def test_real_mode_requires_a_manifest_with_limits():
    """Real hardware must not run on unverified 0-180 fallback limits."""
    with pytest.raises(ValueError, match="hardware_manifest"):
        HandTester(make_config(), simulate=False, speed=1e6)


def test_real_mode_requires_limits_for_every_axis(tmp_path):
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
    config = make_config()
    config["hardware_manifest"] = str(manifest)

    with pytest.raises(ValueError, match="soft_limits"):
        HandTester(config, simulate=False, speed=1e6)


def test_relative_manifest_path_resolves_against_the_config_directory(tmp_path):
    """A config kept beside its manifest must find it via -c."""
    (tmp_path / "hand_config.yaml").write_text("servos: {}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "axes:\n"
        "  - name: index_mcp\n"
        "    role: active\n"
        "    channel: 0\n"
        "    neutral_angle: 95\n"
        "    soft_limits: {min: 85, max: 105}\n",
        encoding="utf-8",
    )
    config = make_config()
    config["hardware_manifest"] = "manifest.yaml"

    tester = HandTester(
        config,
        simulate=True,
        speed=1e6,
        config_path=str(tmp_path / "hand_config.yaml"),
    )
    try:
        assert tester._axis_default("index_mcp") == 95.0
    finally:
        tester.serial.disconnect()
