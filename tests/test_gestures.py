"""Tests for gesture recognition.

Poses are built from the mapper's real angle range (about 158 extended, 68
flexed), not the protocol's 0-180, so the thresholds are exercised against the
numbers the pipeline actually produces.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_mapper.gestures import (
    EXTENDED,
    EXTENDED_MIN,
    FLEXED,
    FLEXED_MAX,
    GESTURES,
    ANY,
    Gesture,
    GestureRecognizer,
    finger_state,
    tip_touch_ratio,
)
from src.hand_mapper.joint_mapper import JointAngles
from src.hand_tracking.mediapipe_tracker import HandLandmarks

OPEN = 155.0
CLOSED = 72.0
LONG = ("index", "middle", "ring", "pinky")


def _angles(**overrides) -> JointAngles:
    angles = JointAngles()
    for name, value in overrides.items():
        setattr(angles, name, value)
    return angles


def _set_finger(angles: JointAngles, finger: str, value: float) -> None:
    setattr(angles, f"{finger}_mcp", value)
    setattr(angles, f"{finger}_pip", value)


def _pose(thumb=OPEN, **finger_values) -> JointAngles:
    """Build a pose from finger -> value, defaulting the rest to open.

    The thumb defaults to open too.  Leaving it at the dataclass default (140)
    would make every pose read as "thumb extended", which is exactly the
    distinction between a fist and a thumbs-up -- so the default has to be
    chosen per pose, not inherited.
    """
    angles = _angles()
    for finger in LONG:
        _set_finger(angles, finger, OPEN)
    for finger, value in finger_values.items():
        _set_finger(angles, finger, value)
    angles.thumb_mcp = thumb
    angles.thumb_ip = thumb
    return angles


def _landmarks(tip_ratio: float, handedness: str = "Right") -> HandLandmarks:
    """Landmarks whose thumb-tip to index-tip distance is ``tip_ratio`` palms."""
    pixels = np.zeros((21, 3), dtype=float)
    for i in range(21):
        pixels[i] = [300.0 + 5.0 * i, 300.0 + 5.0 * i, 0.0]
    pixels[5] = [300.0, 400.0, 0.0]      # index MCP
    pixels[17] = [400.0, 400.0, 0.0]     # pinky MCP -> palm width 100
    pixels[8] = [350.0, 380.0, 0.0]      # index tip
    pixels[4] = [350.0 + tip_ratio * 100.0, 380.0, 0.0]   # thumb tip
    return HandLandmarks(
        landmarks_px=pixels,
        landmarks_norm=pixels / 640.0,
        handedness=handedness,
        handedness_confidence=0.9,
    )


def _settle(recognizer, angles, landmarks=None, frames=None):
    """Feed a pose until the debounce reports it."""
    frames = frames or recognizer.hold_frames
    result = None
    for _ in range(frames):
        result = recognizer.update(angles, landmarks)
    return result


# ---------------------------------------------------------------------------
# Finger state
# ---------------------------------------------------------------------------

def test_a_straight_finger_reads_extended():
    assert finger_state(_pose(), "index") == EXTENDED


def test_a_curled_finger_reads_flexed():
    assert finger_state(_pose(index=CLOSED), "index") == FLEXED


def test_a_half_curled_finger_is_neither():
    """The dead band matters: a fist must not read as an open hand."""
    halfway = (EXTENDED_MIN + FLEXED_MAX) / 2.0

    assert finger_state(_pose(index=halfway), "index") not in (EXTENDED, FLEXED)


def test_the_pip_decides_even_when_the_knuckle_disagrees():
    """A folded middle joint means flexed, whatever the knuckle reads.

    The MCP only moves 20-70 degrees on a real hand (a curled finger points at
    the camera, so its 2D projection collapses).  Requiring it to corroborate
    the PIP left every finger "unclear" during a fist, so fists never matched.
    """
    angles = _pose()
    angles.index_pip = CLOSED

    assert finger_state(angles, "index") == FLEXED


# ---------------------------------------------------------------------------
# Each gesture is recognised from its own pose
# ---------------------------------------------------------------------------

def test_open_hand():
    result = _settle(GestureRecognizer(), _pose())

    assert result.name == "open"


def test_fist():
    """A real fist wraps the thumb across the fingers."""
    pose = _pose(thumb=CLOSED, **{f: CLOSED for f in LONG})

    result = _settle(GestureRecognizer(), pose)

    assert result.name == "fist"


def test_yeah():
    """The user's "yes": index and middle up, ring and pinky down."""
    pose = _pose(index=OPEN, middle=OPEN, ring=CLOSED, pinky=CLOSED)

    result = _settle(GestureRecognizer(), pose)

    assert result.name == "yeah"
    assert result.label == "比耶"


def test_point():
    pose = _pose(index=OPEN, middle=CLOSED, ring=CLOSED, pinky=CLOSED)

    assert _settle(GestureRecognizer(), pose).name == "point"


def test_thumbs_up():
    pose = _pose(**{f: CLOSED for f in LONG})
    pose.thumb_mcp = OPEN
    pose.thumb_ip = OPEN

    assert _settle(GestureRecognizer(), pose).name == "thumbs_up"


def test_ok():
    pose = _pose(index=100.0, middle=OPEN, ring=OPEN, pinky=OPEN)

    result = _settle(GestureRecognizer(), pose, _landmarks(tip_ratio=0.1))

    assert result.name == "ok"


# ---------------------------------------------------------------------------
# Distinctions that must hold
# ---------------------------------------------------------------------------

def test_a_fist_is_not_mistaken_for_thumbs_up():
    """Same four fingers; the thumb is the only difference."""
    pose = _pose(**{f: CLOSED for f in LONG})
    pose.thumb_mcp = CLOSED
    pose.thumb_ip = CLOSED

    assert _settle(GestureRecognizer(), pose).name == "fist"


def test_open_hand_is_not_ok_even_though_the_index_is_unconstrained():
    """OK leaves the index free, so the tip contact is the real discriminator."""
    pose = _pose()

    result = _settle(GestureRecognizer(), pose, _landmarks(tip_ratio=1.5))

    assert result.name == "open"


def test_ok_needs_the_tips_actually_touching():
    pose = _pose(index=100.0, middle=OPEN, ring=OPEN, pinky=OPEN)

    result = _settle(GestureRecognizer(), pose, _landmarks(tip_ratio=1.2))

    assert result.name != "ok"


def test_a_half_closed_hand_matches_nothing():
    halfway = (EXTENDED_MIN + FLEXED_MAX) / 2.0
    pose = _pose(**{f: halfway for f in LONG})

    assert _settle(GestureRecognizer(), pose).matched is False


def test_yeah_is_distinguished_from_point():
    """Both have the index up; the middle finger decides."""
    yeah = _pose(index=OPEN, middle=OPEN, ring=CLOSED, pinky=CLOSED)
    point = _pose(index=OPEN, middle=CLOSED, ring=CLOSED, pinky=CLOSED)

    assert _settle(GestureRecognizer(), yeah).name == "yeah"
    assert _settle(GestureRecognizer(), point).name == "point"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def test_a_decisive_pose_scores_high():
    # 177 is a realistically open hand (measured ~179); 157 is only just past
    # the threshold and is deliberately treated as marginal.
    result = _settle(GestureRecognizer(), _pose(index=177.0, middle=177.0,
                                                ring=177.0, pinky=177.0))

    assert result.name == "open"
    assert result.confidence > 0.7


def test_a_barely_qualifying_pose_scores_low():
    """Just past the threshold is a marginal match, and must say so."""
    barely = EXTENDED_MIN + 1.0
    pose = _pose(**{f: barely for f in LONG})

    result = _settle(GestureRecognizer(), pose)

    assert result.name == "open"
    assert result.confidence < 0.2


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------

def test_a_single_frame_does_not_report_a_gesture():
    """A pose flashing past mid-transition must not be announced."""
    recognizer = GestureRecognizer(hold_frames=3)

    assert recognizer.update(_pose()).matched is False


def test_a_held_pose_is_reported_once_it_settles():
    recognizer = GestureRecognizer(hold_frames=3)
    pose = _pose()

    recognizer.update(pose)
    recognizer.update(pose)
    result = recognizer.update(pose)

    assert result.name == "open"


def test_a_pose_that_does_not_persist_is_never_reported():
    recognizer = GestureRecognizer(hold_frames=3)

    recognizer.update(_pose())
    recognizer.update(_pose(**{f: CLOSED for f in LONG}))
    recognizer.update(_pose())

    assert recognizer.update(_pose()).matched is False


def test_a_settled_gesture_holds_through_a_brief_interruption():
    """A different gesture showing for a frame must not blank a settled one.

    The debounce used to drop the current gesture the instant a different
    gesture appeared, and only report the new one after it had held for
    several frames, so the readout went blank in between.
    """
    recognizer = GestureRecognizer(hold_frames=3)
    _settle(recognizer, _pose())

    result = recognizer.update(_pose(**{f: CLOSED for f in LONG}))

    assert result.name == "open"
    assert result.confidence == 0.0


def test_a_held_new_pose_replaces_the_previous_one():
    recognizer = GestureRecognizer(hold_frames=3)
    _settle(recognizer, _pose())
    closed = _pose(**{f: CLOSED for f in LONG}, thumb=CLOSED)

    for _ in range(2):
        assert recognizer.update(closed).name == "open"
    result = recognizer.update(closed)

    assert result.name == "fist"
    assert result.confidence > 0.0


def test_reset_clears_the_debounce():
    recognizer = GestureRecognizer(hold_frames=2)
    pose = _pose()
    _settle(recognizer, pose)

    recognizer.reset()

    assert recognizer.update(pose).matched is False


def test_a_single_missed_detection_keeps_the_gesture():
    """One dropped frame is not the end of a pose.

    MediaPipe loses the hand for a frame at a time -- 16% of frames in a
    measured capture.  Clearing the debounce on every miss made the readout
    strobe between the gesture and nothing.
    """
    recognizer = GestureRecognizer(hold_frames=2, missing_grace=5)
    _settle(recognizer, _pose())

    result = recognizer.update(None)

    assert recognizer._current == "open"
    assert result.name == "open"
    assert result.label  # the label has to survive the gap too


def test_a_long_gap_clears_the_state():
    recognizer = GestureRecognizer(hold_frames=2, missing_grace=3)
    _settle(recognizer, _pose())

    for _ in range(3):
        recognizer.update(None)

    assert recognizer._current is None


# ---------------------------------------------------------------------------
# Scale invariance and degenerate input
# ---------------------------------------------------------------------------

def test_tip_touch_is_scale_invariant():
    near = _landmarks(tip_ratio=0.2)
    far = HandLandmarks(
        landmarks_px=near.landmarks_px * 0.3,
        landmarks_norm=near.landmarks_norm,
        handedness="Right",
        handedness_confidence=0.9,
    )

    assert tip_touch_ratio(far) == pytest.approx(tip_touch_ratio(near), abs=1e-9)


def test_tip_touch_rejects_degenerate_landmarks():
    collapsed = _landmarks(0.2)
    collapsed.landmarks_px[17] = collapsed.landmarks_px[5]

    assert tip_touch_ratio(collapsed) is None


def test_tip_touch_rejects_none():
    assert tip_touch_ratio(None) is None


def test_missing_angles_is_not_an_error():
    result = GestureRecognizer().update(None)

    assert result.matched is False


def test_fist_matches_when_a_knuckle_reads_above_flexed_max():
    """The MCP must not gate the match.

    A closed pinky's knuckle commonly reads 130-140 degrees, which is above
    FLEXED_MAX.  Demanding that both the PIP and the MCP clear the threshold
    made finger_state() answer "flexed" while _state_margin() answered "no" in
    the same call, so a perfectly good fist matched no gesture at all.
    """
    pose = _pose(index=CLOSED, middle=CLOSED, ring=CLOSED, pinky=CLOSED,
                 thumb=CLOSED)
    pose.pinky_mcp = 133.5
    pose.index_mcp = 91.6

    assert finger_state(pose, "pinky") == FLEXED
    assert _settle(GestureRecognizer(), pose).name == "fist"


def test_non_finite_angles_do_not_match():
    pose = _pose()
    pose.index_mcp = float("nan")

    assert _settle(GestureRecognizer(), pose).matched is False


def test_a_gesture_with_no_constraints_does_not_crash():
    """An all-wildcard spec is degenerate but must not raise."""
    recognizer = GestureRecognizer(
        gestures=(Gesture("anything", "任意", {"index": ANY}),),
        hold_frames=1,
    )

    assert recognizer.update(_pose()).name == "anything"


# ---------------------------------------------------------------------------
# The spec table itself
# ---------------------------------------------------------------------------

def test_gesture_names_are_unique():
    names = [g.name for g in GESTURES]
    assert len(names) == len(set(names))


def test_every_gesture_constrains_something():
    """A gesture with no constraints would match every pose."""
    for gesture in GESTURES:
        has_rule = (
            any(state != ANY for state in gesture.fingers.values())
            or gesture.thumb
            or gesture.tips_touching
        )
        assert has_rule, f"{gesture.name} matches everything"


def test_the_users_yes_gesture_is_present():
    """The user described "yes" as a two-finger V, not a thumbs-up."""
    yeah = next(g for g in GESTURES if g.name == "yeah")

    assert yeah.fingers["index"] == EXTENDED
    assert yeah.fingers["middle"] == EXTENDED
    assert yeah.fingers["ring"] == FLEXED
    assert yeah.fingers["pinky"] == FLEXED
