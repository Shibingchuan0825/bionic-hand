"""
Tests for the JointMapper module.

Validates angle computation from synthetic hand landmarks.
"""

import numpy as np
import pytest

# Add parent to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_mapper.joint_mapper import JointMapper, JointAngles
from src.hand_tracking.mediapipe_tracker import HandLandmarks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_landmarks(points_2d: np.ndarray) -> HandLandmarks:
    """
    Create synthetic HandLandmarks from 2D points (in pixel coordinates).

    Args:
        points_2d: Shape (21, 2) or (21, 3) numpy array of (x, y) or (x, y, z).

    Returns:
        HandLandmarks with the given positions.
    """
    if points_2d.shape[1] == 2:
        # Add z=0 column
        pts_3d = np.hstack([points_2d, np.zeros((21, 1))])
    else:
        pts_3d = points_2d

    return HandLandmarks(
        landmarks_px=pts_3d.astype(np.float32),
        landmarks_norm=pts_3d.astype(np.float32) / 640.0,  # Fake normalization
        handedness="Right",
        handedness_confidence=0.95,
    )


def _flat_pose(value: float) -> JointAngles:
    """Every joint at ``value`` -- a pose with no structure to argue about."""
    pose = JointAngles()
    for field_name in JointAngles.__dataclass_fields__:
        if field_name not in ("confidence", "timestamp"):
            setattr(pose, field_name, value)
    return pose


def make_open_hand_landmarks() -> HandLandmarks:
    """Create synthetic landmarks for a fully open hand."""
    # Simplified coordinate layout
    # Wrist at center-bottom, fingers pointing up
    points = np.array([
        [320, 400, 0],   # 0: Wrist
        [280, 380, 0],   # 1: Thumb CMC
        [240, 350, 0],   # 2: Thumb MCP
        [220, 320, 0],   # 3: Thumb IP
        [200, 280, 0],   # 4: Thumb TIP
        [280, 330, 0],   # 5: Index MCP
        [260, 260, 0],   # 6: Index PIP
        [250, 210, 0],   # 7: Index DIP
        [245, 160, 0],   # 8: Index TIP
        [320, 320, 0],   # 9: Middle MCP
        [320, 240, 0],   # 10: Middle PIP
        [320, 190, 0],   # 11: Middle DIP
        [320, 140, 0],   # 12: Middle TIP
        [360, 330, 0],   # 13: Ring MCP
        [380, 260, 0],   # 14: Ring PIP
        [390, 220, 0],   # 15: Ring DIP
        [395, 180, 0],   # 16: Ring TIP
        [390, 350, 0],   # 17: Pinky MCP
        [420, 300, 0],   # 18: Pinky PIP
        [440, 270, 0],   # 19: Pinky DIP
        [450, 240, 0],   # 20: Pinky TIP
    ], dtype=np.float32)

    return make_landmarks(points)


def make_fist_landmarks() -> HandLandmarks:
    """Create synthetic landmarks for a closed fist."""
    # Fingers curled inward (toward palm, x and y both shift)
    points = np.array([
        [320, 400, 0],   # 0: Wrist
        [290, 385, 0],   # 1: Thumb CMC
        [270, 395, 0],   # 2: Thumb MCP
        [285, 415, 0],   # 3: Thumb IP (curled)
        [300, 425, 0],   # 4: Thumb TIP
        [280, 330, 0],   # 5: Index MCP
        [270, 365, 0],   # 6: Index PIP (curling down + left)
        [255, 390, 0],   # 7: Index DIP
        [245, 410, 0],   # 8: Index TIP
        [310, 320, 0],   # 9: Middle MCP
        [300, 360, 0],   # 10: Middle PIP (curling down + left)
        [285, 390, 0],   # 11: Middle DIP
        [275, 415, 0],   # 12: Middle TIP
        [350, 330, 0],   # 13: Ring MCP
        [345, 365, 0],   # 14: Ring PIP (curling down)
        [335, 390, 0],   # 15: Ring DIP
        [325, 410, 0],   # 16: Ring TIP
        [385, 350, 0],   # 17: Pinky MCP
        [380, 375, 0],   # 18: Pinky PIP (curling down)
        [370, 392, 0],   # 19: Pinky DIP
        [362, 405, 0],   # 20: Pinky TIP
    ], dtype=np.float32)

    return make_landmarks(points)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJointMapper:
    """Tests for the JointMapper class."""

    def test_compute_angles_open_hand(self):
        """Open hand should produce large (extended) angles."""
        mapper = JointMapper(smoothing_enabled=False)
        landmarks = make_open_hand_landmarks()
        angles = mapper.compute_angles(landmarks)

        assert angles is not None
        # Index MCP should be extended (straight finger)
        assert angles.index_mcp > 90, f"Expected extended MCP, got {angles.index_mcp}"
        # Index PIP should also be extended
        assert angles.index_pip > 90, f"Expected extended PIP, got {angles.index_pip}"
        # Middle MCP should be extended
        assert angles.middle_mcp > 90, f"Expected extended MCP, got {angles.middle_mcp}"

    def test_compute_angles_fist(self):
        """Fist should produce smaller (flexed) angles than open hand."""
        mapper = JointMapper(smoothing_enabled=False)
        open_landmarks = make_open_hand_landmarks()
        fist_landmarks = make_fist_landmarks()

        open_angles = mapper.compute_angles(open_landmarks)
        fist_angles = mapper.compute_angles(fist_landmarks)

        assert open_angles is not None
        assert fist_angles is not None

        # PIP joints should be more flexed in fist than open hand
        assert fist_angles.index_pip < open_angles.index_pip, \
            f"Fist PIP ({fist_angles.index_pip:.1f}) should be < Open PIP ({open_angles.index_pip:.1f})"

        assert fist_angles.middle_pip < open_angles.middle_pip, \
            f"Fist PIP ({fist_angles.middle_pip:.1f}) should be < Open PIP ({open_angles.middle_pip:.1f})"

    def test_angle_range(self):
        """All angles should be within [0, 180]."""
        mapper = JointMapper(smoothing_enabled=False)
        landmarks = make_open_hand_landmarks()
        angles = mapper.compute_angles(landmarks)

        assert angles is not None
        for name, value in angles.to_dict().items():
            assert 0 <= value <= 180, \
                f"{name} = {value:.1f} is out of [0, 180] range"

    def test_none_landmarks(self):
        """Passing None should return None."""
        mapper = JointMapper()
        result = mapper.compute_angles(None)
        assert result is None

    def test_to_dict(self):
        """to_dict should return all joint keys."""
        mapper = JointMapper(smoothing_enabled=False)
        landmarks = make_open_hand_landmarks()
        angles = mapper.compute_angles(landmarks)

        assert angles is not None
        d = angles.to_dict()

        expected_keys = {
            "thumb_cmc", "thumb_mcp", "thumb_ip",
            "index_mcp", "index_pip",
            "middle_mcp", "middle_pip",
            "ring_mcp", "ring_pip",
            "pinky_mcp", "pinky_pip",
            "wrist_rot",
        }
        assert set(d.keys()) == expected_keys

    def test_default_angles(self):
        """Default angles should be reasonable open-hand position."""
        angles = JointAngles.default()
        assert angles.index_mcp > 0
        assert angles.index_mcp == 160.0
        assert angles.thumb_mcp == 140.0

    def test_smoothing_stabilizes(self):
        """Smoothing should reduce angle changes between frames."""
        mapper = JointMapper(smoothing_enabled=True, smoothing_alpha=0.2, max_step=5.0)

        # First frame: open hand
        landmarks_open = make_open_hand_landmarks()
        angles1 = mapper.compute_angles(landmarks_open)
        assert angles1 is not None

        # Second frame: fist (big change)
        landmarks_fist = make_fist_landmarks()
        angles2 = mapper.compute_angles(landmarks_fist)
        assert angles2 is not None

        # The smoothed change should be less than the raw difference
        # With alpha=0.2, the second frame should be heavily weighted
        # toward the first frame's values
        raw_diff = abs(angles1.index_pip - angles2.index_pip)
        assert raw_diff < 100, f"Smoothing should reduce large jumps, but diff={raw_diff}"

    def test_invalid_landmarks_are_rejected(self):
        mapper = JointMapper(smoothing_enabled=False)
        landmarks = make_open_hand_landmarks()
        landmarks.landmarks_px[0, 0] = np.inf
        assert not landmarks.is_valid()
        assert mapper.compute_angles(landmarks) is None

    def test_collapsed_landmarks_are_rejected(self):
        mapper = JointMapper(smoothing_enabled=False)
        points = np.zeros((21, 3), dtype=np.float32)
        landmarks = make_landmarks(points)
        assert not landmarks.is_valid()
        assert mapper.compute_angles(landmarks) is None

        """Angle between identical points should not fail."""
        mapper = JointMapper()
        # Three identical points → degenerate case
        p = np.array([100.0, 100.0, 0.0])
        angle = mapper._angle_between(p, p, p)
        assert angle == 180.0  # Fully extended (degenerate case)

    def test_angle_between_straight_line(self):
        """Angle on a straight line should be 180°."""
        mapper = JointMapper()
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([50.0, 0.0, 0.0])
        c = np.array([100.0, 0.0, 0.0])
        angle = mapper._angle_between(a, b, c)
        assert angle == pytest.approx(180.0, abs=1.0)

    def test_angle_between_right_angle(self):
        """Right angle should be 90°."""
        mapper = JointMapper()
        b = np.array([0.0, 0.0, 0.0])
        a = np.array([100.0, 0.0, 0.0])   # ← INTO joint
        c = np.array([0.0, 100.0, 0.0])   # ↑ OUT of joint
        angle = mapper._angle_between(a, b, c)
        assert angle == pytest.approx(90.0, abs=1.0)

    def test_rate_limit_is_not_compounded_by_the_ema(self):
        """max_step means degrees per frame, not degrees before the blend.

        Capping the target before the EMA multiplied the two filters: a 5
        degree cap followed by a 0.3 blend moved 1.5 degrees per frame, so a
        100 degree pose change needed 67 frames -- 2.2 s at 30 fps -- to
        arrive.  Filtering first leaves the cap as the only rate limit.
        """
        mapper = JointMapper(smoothing_enabled=True,
                             smoothing_alpha=0.3, max_step=5.0)

        mapper._prev_angles = _flat_pose(180.0)
        out = mapper._apply_constraints(_flat_pose(60.0),
                                        make_open_hand_landmarks())

        # 180 -> 175, not 180 -> 178.5.
        assert out.index_pip == pytest.approx(175.0, abs=0.01)

    def test_instant_angles_are_not_filtered(self):
        """Recognition reads the unfiltered pose; the actuator path does not."""
        mapper = JointMapper(smoothing_enabled=True,
                             smoothing_alpha=0.1, max_step=2.0)

        for _ in range(5):
            mapper.compute_angles(make_open_hand_landmarks())
        smoothed = mapper.compute_angles(make_fist_landmarks())

        assert mapper.instant_angles is not None
        assert abs(mapper.instant_angles.index_pip - smoothed.index_pip) > 5.0

    def test_wrist_rotation_survives_a_mislabelled_hand(self):
        """A mirrored camera makes MediaPipe call a right hand "Left".

        Trusting the label flipped the reference direction, the roll came out
        near -180, and the +90 offset clipped the result to 0 -- for every
        frame of a whole measured capture, so the wrist axis was simply dead.
        """
        landmarks = make_open_hand_landmarks()
        landmarks.handedness = "Left"

        unforced = JointMapper(smoothing_enabled=False).compute_angles(landmarks)
        forced = JointMapper(smoothing_enabled=False,
                             forced_handedness="Right").compute_angles(landmarks)

        assert unforced.wrist_rot == 0.0
        assert 20.0 < forced.wrist_rot < 160.0

    def test_reset_smoothing(self):
        """Reset should clear accumulated smoothing state."""
        mapper = JointMapper(smoothing_enabled=True, smoothing_alpha=0.5)
        landmarks = make_open_hand_landmarks()

        # Process one frame
        angles1 = mapper.compute_angles(landmarks)
        assert angles1 is not None

        # Reset
        mapper.reset_smoothing()

        # After reset, next frame should have full effect (no smoothing from previous)
        angles2 = mapper.compute_angles(landmarks)
        assert angles2 is not None
        # Should be nearly identical since no smoothing accumulated
        assert angles2.index_mcp == pytest.approx(angles1.index_mcp, abs=0.1)


class TestJointAngles:
    """Tests for the JointAngles dataclass."""

    def test_from_dict(self):
        """from_dict should create JointAngles from a dictionary."""
        d = {"thumb_cmc": 100.0, "thumb_mcp": 110.0, "thumb_ip": 120.0,
             "index_mcp": 130.0, "index_pip": 140.0,
             "middle_mcp": 130.0, "middle_pip": 140.0,
             "ring_mcp": 130.0, "ring_pip": 140.0,
             "pinky_mcp": 130.0, "pinky_pip": 140.0,
             "wrist_rot": 90.0}
        angles = JointAngles.from_dict(d)
        assert angles.thumb_cmc == 100.0
        assert angles.index_mcp == 130.0


class TestWristRotation:
    """wrist_rot must measure palm roll about the forearm, not image tilt."""

    @staticmethod
    def _rolled_hand(roll_deg, *, tilt_deg=0.0, handedness="Right"):
        """Synthetic hand whose palm is rolled and which is tilted in frame.

        The hand points up the image; ``roll_deg`` rotates the palm about that
        long axis, and ``tilt_deg`` rotates the whole hand (forearm included)
        in the image plane.
        """
        roll = np.radians(roll_deg)
        tilt = np.radians(tilt_deg)

        forearm = np.array([0.0, -100.0])
        perp = np.array([-forearm[1], forearm[0]])
        perp = perp / np.linalg.norm(perp)
        lateral = np.array([
            perp[0] * np.cos(roll) - perp[1] * np.sin(roll),
            perp[0] * np.sin(roll) + perp[1] * np.cos(roll),
        ])

        points = np.zeros((21, 3))
        points[9, :2] = forearm
        points[5, :2] = -lateral * 30.0   # index side
        points[17, :2] = lateral * 30.0   # pinky side
        for index in range(1, 21):
            if index not in (5, 9, 17):
                points[index, :2] = forearm * (index / 20.0)

        if handedness == "Left":
            points[:, 0] *= -1.0

        rotate = np.array([
            [np.cos(tilt), -np.sin(tilt)],
            [np.sin(tilt), np.cos(tilt)],
        ])
        points[:, :2] = points[:, :2] @ rotate.T + np.array([320.0, 400.0])

        return HandLandmarks(
            landmarks_px=points.astype(np.float32),
            landmarks_norm=(points / 640.0).astype(np.float32),
            handedness=handedness,
            handedness_confidence=0.95,
        )

    @staticmethod
    def _wrist_rot(landmarks):
        mapper = JointMapper(smoothing_enabled=False, max_step=1000.0)
        return mapper.compute_angles(landmarks).wrist_rot

    def test_neutral_palm_reads_as_ninety(self):
        assert self._wrist_rot(self._rolled_hand(0.0)) == pytest.approx(90.0, abs=1.0)

    def test_roll_direction_is_signed(self):
        """Pronation and supination must differ, not collapse to one value."""
        positive = self._wrist_rot(self._rolled_hand(30.0))
        negative = self._wrist_rot(self._rolled_hand(-30.0))

        assert positive == pytest.approx(60.0, abs=2.0)
        assert negative == pytest.approx(120.0, abs=2.0)

    def test_roll_is_symmetric_about_neutral(self):
        positive = self._wrist_rot(self._rolled_hand(30.0))
        negative = self._wrist_rot(self._rolled_hand(-30.0))

        assert (positive + negative) / 2.0 == pytest.approx(90.0, abs=1.0)

    def test_left_hand_mirrors_the_right_hand(self):
        """An image-mirrored hand must report the same wrist angle."""
        right = self._wrist_rot(self._rolled_hand(30.0, handedness="Right"))
        left = self._wrist_rot(
            self._rolled_hand(30.0, handedness="Left")
        )

        assert left == pytest.approx(right, abs=1.0)

    def test_absolute_image_tilt_does_not_move_the_wrist_angle(self):
        """Moving the whole arm must not look like twisting the wrist."""
        upright = self._wrist_rot(self._rolled_hand(0.0, tilt_deg=0.0))
        tilted = self._wrist_rot(self._rolled_hand(0.0, tilt_deg=37.0))

        assert tilted == pytest.approx(upright, abs=1.0)

    def test_tilt_does_not_change_a_rolled_wrist_either(self):
        for tilt in (0.0, 30.0, 90.0, 150.0):
            assert self._wrist_rot(
                self._rolled_hand(25.0, tilt_deg=tilt)
            ) == pytest.approx(65.0, abs=2.0)

    def test_a_flat_landmark_set_is_not_a_crash(self):
        """Degenerate geometry must return the neutral value, not raise."""
        pixels = np.zeros((21, 3), dtype=np.float32)
        degenerate = HandLandmarks(
            landmarks_px=pixels,
            landmarks_norm=pixels,
            handedness="Right",
            handedness_confidence=0.95,
        )
        mapper = JointMapper(smoothing_enabled=False, max_step=1000.0)

        assert mapper._compute_wrist_rotation(degenerate) == pytest.approx(90.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
