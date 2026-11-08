"""Tests for dynamic (motion) gesture recognition.

Paths are generated geometrically rather than recorded from a camera, so the
circle cases are exact: a sampled circle has a known direction and a known
path-to-displacement ratio, which is what the classifier keys on.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_mapper.motion_gestures import (
    CLOCKWISE,
    COUNTER_CLOCKWISE,
    MOVE,
    STOP,
    UNKNOWN,
    MotionRecognizer,
    classify_path,
)
from src.hand_tracking.mediapipe_tracker import HandLandmarks


def circle(radius=1.0, turns=1.0, direction=1, samples=24, centre=(0.0, 0.0)):
    """Sample a circle.  direction=+1 is counter-clockwise in image coords."""
    angles = np.linspace(0.0, 2 * math.pi * turns, samples, endpoint=False)
    return [(centre[0] + radius * math.cos(a) * direction,
             centre[1] + radius * math.sin(a)) for a in angles]


def straight(length=2.0, samples=20, start=(0.0, 0.0), end=(2.0, 0.0)):
    xs = np.linspace(start[0], end[0], samples)
    ys = np.linspace(start[1], end[1], samples)
    return list(zip(xs, ys))


def still(samples=20):
    return [(0.0, 0.0)] * samples


# ---------------------------------------------------------------------------
# The four classes
# ---------------------------------------------------------------------------

def test_a_circle_clockwise_is_recognised():
    # In image coordinates y grows downward, so a clockwise screen motion is a
    # negative shoelace area.
    result = classify_path(circle(radius=1.0, direction=-1))

    assert result.name == CLOCKWISE
    assert result.confidence > 0.5


def test_a_circle_counter_clockwise_is_recognised():
    result = classify_path(circle(radius=1.0, direction=+1))

    assert result.name == COUNTER_CLOCKWISE
    assert result.confidence > 0.5


def test_a_straight_motion_is_a_move():
    result = classify_path(straight())

    assert result.name == MOVE
    assert result.displacement_palms > 1.0


def test_no_motion_is_a_stop():
    result = classify_path(still())

    assert result.name == STOP


def test_two_full_turns_still_reads_as_a_loop():
    result = classify_path(circle(radius=1.0, turns=2.0, samples=40))

    assert result.name in (CLOCKWISE, COUNTER_CLOCKWISE)


# ---------------------------------------------------------------------------
# Distinctions that must hold
# ---------------------------------------------------------------------------

def test_direction_is_not_confused():
    """The two circle directions must never swap."""
    cw = classify_path(circle(radius=1.0, direction=-1))
    ccw = classify_path(circle(radius=1.0, direction=+1))

    assert cw.name != ccw.name
    assert cw.signed_area * ccw.signed_area < 0


def test_a_small_wiggle_is_not_a_loop():
    """Movement too small to be intentional must not fire a gesture."""
    result = classify_path(circle(radius=0.05))

    assert result.name != CLOCKWISE
    assert result.name != COUNTER_CLOCKWISE


def test_a_loop_off_centre_is_still_a_loop():
    """The area is taken about the path's own centroid, so position is irrelevant."""
    result = classify_path(circle(radius=1.0, direction=+1, centre=(5.0, -3.0)))

    assert result.name == COUNTER_CLOCKWISE


def test_scale_invariance():
    """The same shape at a different size classifies the same way.

    Normalisation to palm widths happens before this function, so a path is
    already dimensionless; this checks the classifier does not reintroduce a
    size dependence.
    """
    small = classify_path(circle(radius=0.8, direction=+1))
    large = classify_path(circle(radius=2.5, direction=+1))

    assert small.name == large.name == COUNTER_CLOCKWISE


def test_too_few_points_is_unknown():
    assert classify_path([(0.0, 0.0), (1.0, 0.0)]).name == UNKNOWN


def test_non_finite_points_are_rejected():
    assert classify_path([(0, 0), (float("nan"), 0), (1, 1), (2, 2)]).name == UNKNOWN


# ---------------------------------------------------------------------------
# The stateful wrapper
# ---------------------------------------------------------------------------

def _landmarks_at(x: float, y: float, palm_width: float = 100.0) -> HandLandmarks:
    pixels = np.zeros((21, 3), dtype=float)
    for i in range(21):
        pixels[i] = [x, y, 0.0]
    pixels[5] = [x, y, 0.0]
    pixels[17] = [x + palm_width, y, 0.0]
    pixels[8] = [x, y, 0.0]          # index tip follows the argument
    return HandLandmarks(
        landmarks_px=pixels,
        landmarks_norm=pixels / 640.0,
        handedness="Right",
        handedness_confidence=0.9,
    )


def _feed(recognizer, points, palm_width=100.0):
    result = None
    for x, y in points:
        result = recognizer.update(_landmarks_at(x, y, palm_width))
    return result


def test_the_wrapper_reports_nothing_until_the_window_fills():
    recognizer = MotionRecognizer(window=10)

    for _ in range(5):
        result = recognizer.update(_landmarks_at(0.0, 0.0))

    assert result.matched is False


def test_the_wrapper_names_a_traced_circle():
    recognizer = MotionRecognizer(window=12)
    # One palm width = 100 px, so a radius of 1 palm traces a real circle.
    path = [(cx * 100.0, cy * 100.0) for cx, cy in circle(radius=1.0, direction=+1,
                                                          samples=12)]

    result = _feed(recognizer, path)

    assert result.name == COUNTER_CLOCKWISE


def test_the_wrapper_is_invariant_to_hand_distance():
    """The same motion twice as far from the camera must classify the same.

    Both the path and the palm width scale together, so the normalised shape is
    unchanged -- which is the whole reason for dividing by palm width.
    """
    near = MotionRecognizer(window=12)
    far = MotionRecognizer(window=12)
    path = [(cx * 100.0, cy * 100.0) for cx, cy in circle(radius=1.0, direction=+1,
                                                          samples=12)]

    near_result = _feed(near, path, palm_width=100.0)
    far_result = _feed(far, [(x * 2, y * 2) for x, y in path], palm_width=200.0)

    assert near_result.name == far_result.name == COUNTER_CLOCKWISE


def test_losing_the_hand_clears_the_trail():
    recognizer = MotionRecognizer(window=10)
    _feed(recognizer, circle(radius=1.0, samples=12))

    recognizer.update(None)

    assert len(recognizer._path) == 0


def test_reset_clears_the_trail():
    recognizer = MotionRecognizer(window=10)
    _feed(recognizer, circle(radius=1.0, samples=12))

    recognizer.reset()

    assert len(recognizer._path) == 0


def test_a_still_hand_reads_stop_once_the_window_fills():
    recognizer = MotionRecognizer(window=10)

    result = _feed(recognizer, [(50.0, 50.0)] * 12)

    assert result.name == STOP
