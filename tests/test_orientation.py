"""Tests for hand-orientation detection.

Synthetic landmarks are built to a *specified* normalised palm area rather than
hand-placed, so each case is unambiguous: the geometry is solved backwards from
the number the detector is supposed to recover.  That makes the edge-on and
hysteresis cases exact instead of approximate.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_mapper.orientation import (
    ENTER_SIDE,
    EXIT_SIDE,
    FACING_BACK,
    FACING_FRONT,
    FACING_SIDE,
    FACING_UNKNOWN,
    HandOrientation,
    OrientationDetector,
    orientation_is_confident,
    palm_signed_area,
)
from src.hand_tracking.mediapipe_tracker import HandLandmarks

WRIST_PX = (320.0, 480.0)
#: A palm edge pointing up and to the left.  Its length is irrelevant: the
#: measure is normalised, which is the property being relied on.
V1 = (-40.0, -100.0)


def _landmarks_for_area(signed: float, handedness: str = "Right") -> HandLandmarks:
    """Build landmarks whose palm triangle has the given normalised area.

    The measure is ``sin(theta)`` for the angle between the wrist->index and
    wrist->pinky edges, so rotating the second edge by ``asin(signed)`` produces
    exactly the requested value.
    """
    signed = max(-1.0, min(1.0, signed))
    theta = math.asin(signed)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    v2 = (V1[0] * cos_t - V1[1] * sin_t, V1[0] * sin_t + V1[1] * cos_t)

    pixels = np.zeros((21, 3), dtype=float)
    for i in range(21):
        pixels[i] = [WRIST_PX[0] + 5.0 * i, WRIST_PX[1] - 8.0 * i, 0.0]
    pixels[0] = [WRIST_PX[0], WRIST_PX[1], 0.0]
    pixels[5] = [WRIST_PX[0] + V1[0], WRIST_PX[1] + V1[1], 0.0]
    pixels[17] = [WRIST_PX[0] + v2[0], WRIST_PX[1] + v2[1], 0.0]

    return HandLandmarks(
        landmarks_px=pixels,
        landmarks_norm=pixels / 640.0,
        handedness=handedness,
        handedness_confidence=0.9,
    )


# ---------------------------------------------------------------------------
# The raw measure
# ---------------------------------------------------------------------------

def test_signed_area_recovers_the_constructed_value():
    for target in (-0.9, -0.5, 0.0, 0.5, 0.9):
        measured = palm_signed_area(_landmarks_for_area(target))

        assert measured == pytest.approx(target, abs=1e-9)


def test_signed_area_is_scale_invariant():
    """A far-away hand must read the same as a near one."""
    near = _landmarks_for_area(0.7)
    far = HandLandmarks(
        landmarks_px=near.landmarks_px * 0.25,
        landmarks_norm=near.landmarks_norm,
        handedness="Right",
        handedness_confidence=0.9,
    )

    assert palm_signed_area(far) == pytest.approx(palm_signed_area(near), abs=1e-9)


def test_signed_area_rejects_degenerate_geometry():
    collapsed = _landmarks_for_area(0.7)
    collapsed.landmarks_px[5] = collapsed.landmarks_px[0]      # index at the wrist

    assert palm_signed_area(collapsed) is None


def test_signed_area_rejects_non_finite_points():
    broken = _landmarks_for_area(0.7)
    broken.landmarks_px[5] = [float("nan"), 0.0, 0.0]

    assert palm_signed_area(broken) is None


def test_signed_area_rejects_none():
    assert palm_signed_area(None) is None


# ---------------------------------------------------------------------------
# Broadside classification
# ---------------------------------------------------------------------------

def test_right_hand_back_toward_camera():
    detector = OrientationDetector()

    result = detector.update(_landmarks_for_area(0.9, "Right"))

    assert result.facing == FACING_BACK
    assert result.confidence > 0.5


def test_right_hand_palm_toward_camera():
    detector = OrientationDetector()

    result = detector.update(_landmarks_for_area(-0.9, "Right"))

    assert result.facing == FACING_FRONT


def test_left_hand_mirrors_the_right():
    """The same physical pose reads the same way whichever hand makes it.

    Turning a left hand over produces the mirrored image, so the sign of the
    palm area flips with it; a detector that ignored handedness would call the
    back of a left hand "palm".
    """
    back_left = OrientationDetector().update(_landmarks_for_area(-0.9, "Left"))
    front_left = OrientationDetector().update(_landmarks_for_area(0.9, "Left"))

    assert back_left.facing == FACING_BACK
    assert front_left.facing == FACING_FRONT


def test_invert_swaps_front_and_back():
    """Calibration escape hatch, for a camera whose image is mirrored."""
    detector = OrientationDetector(invert_front_back=True)

    assert detector.update(_landmarks_for_area(0.9)).facing == FACING_FRONT


# ---------------------------------------------------------------------------
# The edge-on band
# ---------------------------------------------------------------------------

def test_edge_on_is_recognised():
    detector = OrientationDetector()

    result = detector.update(_landmarks_for_area(0.0))

    assert result.facing == FACING_SIDE


def test_a_clearly_edge_on_reading_is_confident_about_being_edge_on():
    """Edge-on is a reliable statement about the pose.

    The geometry is unambiguous that the hand is side-on -- what it cannot say
    is which way the palm would face if the hand turned slightly.  So a clear
    edge-on reading is confident, not uncertain.
    """
    detector = OrientationDetector()

    result = detector.update(_landmarks_for_area(0.0))

    assert result.facing == FACING_SIDE
    assert orientation_is_confident(result) is True


def test_a_reading_on_the_decision_boundary_is_not_confident():
    """This is the genuinely uncertain case: side and broadside look alike.

    Confidence is the distance from the boundary, so it bottoms out here rather
    than at either extreme.
    """
    detector = OrientationDetector()

    result = detector.update(_landmarks_for_area(ENTER_SIDE))

    assert result.confidence < 0.2
    assert orientation_is_confident(result) is False


def test_broadside_readings_are_confident():
    detector = OrientationDetector()

    result = detector.update(_landmarks_for_area(0.95))

    assert orientation_is_confident(result) is True


def test_hysteresis_prevents_flicker_at_the_boundary():
    """A hand hovering at the boundary must not alternate between states.

    Values between ENTER_SIDE and EXIT_SIDE mean different things depending on
    which state the detector is already in.  That gap is the whole point.
    """
    assert ENTER_SIDE < EXIT_SIDE
    between = (ENTER_SIDE + EXIT_SIDE) / 2.0

    detector = OrientationDetector()
    # Start broadside, then drift into the dead-band from above.
    assert detector.update(_landmarks_for_area(0.9)).facing == FACING_BACK
    assert detector.update(_landmarks_for_area(between)).facing == FACING_BACK

    # Dip clearly edge-on, then come back into the dead-band: it must stay side.
    assert detector.update(_landmarks_for_area(0.0)).facing == FACING_SIDE
    assert detector.update(_landmarks_for_area(between)).facing == FACING_SIDE

    # Only a clearly broadside reading pulls it back out.
    assert detector.update(_landmarks_for_area(0.9)).facing == FACING_BACK


def test_reset_clears_the_hysteresis_state():
    detector = OrientationDetector()
    detector.update(_landmarks_for_area(0.0))

    detector.reset()

    assert detector._in_side is False


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------

def test_missing_hand_is_unknown_not_an_error():
    result = OrientationDetector().update(None)

    assert result.facing == FACING_UNKNOWN
    assert orientation_is_confident(result) is False


def test_degenerate_geometry_is_unknown():
    collapsed = _landmarks_for_area(0.9)
    collapsed.landmarks_px[17] = collapsed.landmarks_px[0]

    result = OrientationDetector().update(collapsed)

    assert result.facing == FACING_UNKNOWN


def test_a_dropped_frame_does_not_disturb_the_hysteresis_state():
    """A missed detection must not silently reset the band."""
    detector = OrientationDetector()
    detector.update(_landmarks_for_area(0.0))          # now in "side"

    detector.update(None)                              # hand lost for a frame

    assert detector._in_side is True


def test_default_result_is_unknown():
    assert HandOrientation().facing == FACING_UNKNOWN
