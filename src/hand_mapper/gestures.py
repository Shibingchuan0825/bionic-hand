"""
Gesture recognition from the solved joint angles.

A gesture is a named pose in the same 12-dimensional angle space the rest of the
pipeline already speaks.  That is deliberate: the same definition serves two
purposes at once --

* **recognition** -- "the operator is making this shape now", and
* **target pose** -- "drive the hand to this shape",

so a recognised gesture can be handed straight to the hardware without a second
description of the same pose existing anywhere.

Why rules rather than a trained classifier
------------------------------------------
The angle features are already scale-invariant (they are interior angles, so a
large hand and a small one produce the same numbers), which removes the main
reason a classifier would be needed.  Rules are also inspectable: when a gesture
misreads you can see which threshold failed, and tune that one number.  A
classifier would need training data this project does not have and would fail
in ways that are much harder to diagnose.

The cost is brittleness to unusual hand shapes, which is why each match reports
a confidence: a pose that only just satisfies the rules is flagged as marginal
rather than presented as a certainty.

Adding a gesture
----------------
Append a :class:`Gesture` to :data:`GESTURES`.  Order matters -- the first match
wins -- so put more specific shapes earlier.  See the module tests for the
convention used to describe one.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..hand_tracking.mediapipe_tracker import HandLandmarks
from .joint_mapper import JointAngles

logger = logging.getLogger(__name__)

# Per-finger state, and the wildcard for "this finger does not matter here".
EXTENDED = "extended"
FLEXED = "flexed"
ANY = "any"
UNCLEAR = "unclear"

# Thresholds applied to the PIP angle.
#
# The PIP is used as the primary signal, not the MCP.  Measured on a real hand
# (tools/measure_tracking.py): the PIPs swing over 65-155 degrees, while the
# MCPs move only 20-70 -- a curled finger points at the camera and its 2D
# projection collapses, so the knuckle angle is both small in range and noisy.
# Classifying on the MCP made every finger "unclear" during a fist, so the fist
# gesture never matched at all.
#
# Measured PIP ranges: index 115-180, middle 112-180, ring 26-179, pinky 44-180.
EXTENDED_MIN = 150.0
FLEXED_MAX = 120.0
RANGE_FLEXED = 30.0
RANGE_EXTENDED = 180.0

# The thumb needs its own thresholds.  Measured over a 15 s open/fist capture
# (tools/measure_tracking.py): thumb_mcp 144-180, thumb_ip 122-180.  Reusing
# the finger thresholds put "extended" at 150 on both axes, which the thumb
# only clears at the very end of its travel, and put "flexed" at 120, which a
# closed thumb never reaches -- so the thumb read as unclear in both
# directions and thumbs_up kept falling through to fist.
THUMB_EXTENDED_MIN = 150.0
THUMB_FLEXED_MAX = 135.0

# Tip-to-tip distance for a touch gesture, as a fraction of palm width.  Palm
# width is used as the scale reference so the rule does not depend on how far
# the hand is from the camera.
TIP_TOUCH_RATIO = 0.40

# Frames a pose must persist before it is reported.  Stops the readout strobing
# through a transition.
DEFAULT_HOLD_FRAMES = 3

# Frames the hand may be missing before the current gesture is dropped.
#
# MediaPipe loses a hand for a frame or two at a time -- 16% of frames in a
# measured 15 s capture.  Resetting the debounce on every miss made the readout
# alternate between the gesture and nothing, which reads as "recognition is
# unreliable" even when the classifier was correct on every frame it saw.
DEFAULT_MISSING_GRACE = 5

THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
PINKY_MCP = 17

LONG_FINGERS = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class Gesture:
    """A named pose, described as constraints on the joint angles."""

    name: str
    label: str
    #: finger -> required state (EXTENDED / FLEXED / ANY)
    fingers: Mapping[str, str]
    #: thumb axis -> required state
    thumb: Mapping[str, str] = field(default_factory=dict)
    #: two digits whose tips must be touching, e.g. ("thumb", "index")
    tips_touching: Optional[Tuple[str, str]] = None


@dataclass
class GestureResult:
    """Outcome of one recognition attempt."""

    name: Optional[str] = None
    label: str = ""
    #: 0..1.  How decisively the pose matched, not merely that it matched.
    confidence: float = 0.0

    @property
    def matched(self) -> bool:
        return self.name is not None


# Ordered most specific first: the first match wins.
GESTURES: Tuple[Gesture, ...] = (
    Gesture(
        name="ok",
        label="OK",
        # The index curls into the circle, so its own state is not constrained;
        # the tip contact is what distinguishes this from an open hand.
        fingers={"index": ANY, "middle": EXTENDED, "ring": EXTENDED, "pinky": EXTENDED},
        tips_touching=("thumb", "index"),
    ),
    Gesture(
        name="yeah",
        label="比耶",
        fingers={"index": EXTENDED, "middle": EXTENDED, "ring": FLEXED, "pinky": FLEXED},
    ),
    Gesture(
        name="point",
        label="指向",
        fingers={"index": EXTENDED, "middle": FLEXED, "ring": FLEXED, "pinky": FLEXED},
    ),
    Gesture(
        name="thumbs_up",
        label="点赞",
        fingers={f: FLEXED for f in LONG_FINGERS},
        # The discriminator against a fist is the thumb: straight here, wrapped
        # there.  The CMC axis is left free because its reading depends on how
        # far the thumb is splayed sideways, which varies between people.
        thumb={"thumb_mcp": EXTENDED, "thumb_ip": EXTENDED},
    ),
    Gesture(
        name="fist",
        label="握拳",
        fingers={f: FLEXED for f in LONG_FINGERS},
        # The thumb is what separates a fist from a thumbs-up, so it has to be
        # constrained here too.  Without it a thumbs-up whose thumb read a few
        # degrees short of "extended" fell through to this gesture and was
        # reported as a fist.
        thumb={"thumb_ip": FLEXED},
    ),
    Gesture(
        name="open",
        label="张开",
        fingers={f: EXTENDED for f in LONG_FINGERS},
    ),
)


def per_finger_margins(
    angles: JointAngles,
) -> Dict[str, Dict[str, Optional[float]]]:
    """How decisively each finger is extended/flexed, in 0..1.

    Returns ``{finger: {EXTENDED: margin, FLEXED: margin}}``.  A margin of 1
    means "fully in that state"; 0 means "at the threshold boundary"; None means
    invalid.

    Used by the gesture-assist driver to smoothly blend between the raw tracking and a
    clean preset pose on a per-finger basis: a finger that is clearly extended
    rides the preset extended target (low noise, while an ambiguous finger stays live.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for finger in LONG_FINGERS:
        value = _axis(angles, f"{finger}_pip")
        out[finger] = {
            EXTENDED: _state_margin(value, EXTENDED) if value is not None else None,
            FLEXED: _state_margin(value, FLEXED) if value is not None else None,
        }
    # Thumb uses its own thresholds and its own margins
    thumb_mcp = _axis(angles, "thumb_mcp")
    thumb_ip = _axis(angles, "thumb_ip")
    out["thumb"] = {
        EXTENDED: min(
            _thumb_margin(thumb_mcp, EXTENDED) or 0.0,
        _thumb_margin(thumb_ip, EXTENDED) or 0.0,
    ) if (thumb_mcp is not None and thumb_ip is not None) else None,
    FLEXED: min(
        _thumb_margin(thumb_mcp, FLEXED) or 0.0,
        _thumb_margin(thumb_ip, FLEXED) or 0.0,
    ) if (thumb_mcp is not None and thumb_ip is not None) else None,
}
    return out


def _axis(angles: JointAngles, name: str) -> Optional[float]:
    value = getattr(angles, name, None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def finger_state(angles: JointAngles, finger: str) -> str:
    """Classify one finger as extended, flexed, or neither.

    The PIP decides.  The MCP is deliberately not required to agree: its
    measured range is only 20-70 degrees, so demanding it reach the same
    threshold as the PIP means no finger is ever classified -- which is exactly
    why fists stopped being recognised.  The PIP swings over 65-155 degrees and
    is the reliable signal.
    """
    pip = _axis(angles, f"{finger}_pip")
    if pip is None:
        return UNCLEAR
    if pip >= EXTENDED_MIN:
        return EXTENDED
    if pip <= FLEXED_MAX:
        return FLEXED
    return UNCLEAR


def tip_touch_ratio(landmarks: Optional[HandLandmarks]) -> Optional[float]:
    """Distance between the thumb and index tips, over palm width.

    Returns ``None`` when the landmarks are missing or degenerate.  The value is
    scale-invariant, so a distant hand reads the same as a close one.
    """
    if landmarks is None:
        return None
    points = getattr(landmarks, "landmarks_px", None)
    if points is None:
        return None
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] <= PINKY_MCP or points.shape[1] < 2:
        return None

    thumb_tip, index_tip = points[THUMB_TIP][:2], points[INDEX_TIP][:2]
    palm_a, palm_b = points[INDEX_MCP][:2], points[PINKY_MCP][:2]
    if not all(np.all(np.isfinite(p)) for p in (thumb_tip, index_tip, palm_a, palm_b)):
        return None

    palm_width = float(np.linalg.norm(palm_b - palm_a))
    if palm_width < 1e-6:
        return None
    return float(np.linalg.norm(index_tip - thumb_tip)) / palm_width


def _frame_is_finite(angles: JointAngles) -> bool:
    """Whether every angle in the frame is a real number.

    A NaN or infinity anywhere means the geometry for that frame is broken --
    MediaPipe lost a joint, or a normalisation divided by zero.  Classification
    now leans on the PIP, so a broken MCP alone would no longer block a match;
    refusing the whole frame keeps the original safety property, and stops a
    partly-broken frame from being reported as a confident pose.
    """
    for field_name in JointAngles.__dataclass_fields__:
        if field_name in ("confidence", "timestamp"):
            continue
        value = _axis(angles, field_name)
        if value is None:
            return False
    return True


def _thumb_margin(value: float, required: str) -> Optional[float]:
    """How decisively a thumb axis satisfies a required state, in 0..1."""
    if required == ANY:
        return 1.0
    if required == EXTENDED:
        if value < THUMB_EXTENDED_MIN:
            return None
        return min(1.0, (value - THUMB_EXTENDED_MIN)
                   / max(180.0 - THUMB_EXTENDED_MIN, 1e-6))
    if required == FLEXED:
        if value > THUMB_FLEXED_MAX:
            return None
        return min(1.0, (THUMB_FLEXED_MAX - value)
                   / max(THUMB_FLEXED_MAX - 90.0, 1e-6))
    return None


def _state_margin(value: float, required: str) -> Optional[float]:
    """How decisively a value satisfies a required state, in 0..1.

    0 means it only just qualifies; 1 means it is at the far end of the range.
    ``None`` means it does not qualify at all.
    """
    if required == ANY:
        return 1.0
    if required == EXTENDED:
        if value < EXTENDED_MIN:
            return None
        span = max(RANGE_EXTENDED - EXTENDED_MIN, 1e-6)
        return min(1.0, (value - EXTENDED_MIN) / span)
    if required == FLEXED:
        if value > FLEXED_MAX:
            return None
        span = max(FLEXED_MAX - RANGE_FLEXED, 1e-6)
        return min(1.0, (FLEXED_MAX - value) / span)
    return None


class GestureRecognizer:
    """Recognise a static gesture from joint angles.

    Stateful only for the debounce: a pose must persist for a few frames before
    it is reported, so the readout does not strobe while the hand is moving
    between shapes.

    Usage::

        recognizer = GestureRecognizer()
        result = recognizer.update(angles, landmarks)
        if result.matched:
            ...  # result.name, result.confidence
    """

    def __init__(
        self,
        gestures: Tuple[Gesture, ...] = GESTURES,
        *,
        hold_frames: int = DEFAULT_HOLD_FRAMES,
        tip_touch_ratio: float = TIP_TOUCH_RATIO,
        missing_grace: int = DEFAULT_MISSING_GRACE,
    ):
        self.gestures = tuple(gestures)
        self.hold_frames = max(1, int(hold_frames))
        self.tip_touch_ratio = float(tip_touch_ratio)
        self.missing_grace = max(1, int(missing_grace))
        self._candidate: Optional[str] = None
        self._candidate_frames = 0
        self._current: Optional[str] = None
        self._current_label: str = ""
        self._missing_frames = 0

    def reset(self) -> None:
        """Forget the debounce state (e.g. after the hand leaves the frame)."""
        self._candidate = None
        self._candidate_frames = 0
        self._current = None
        self._current_label = ""
        self._missing_frames = 0

    def update(
        self,
        angles: Optional[JointAngles],
        landmarks: Optional[HandLandmarks] = None,
    ) -> GestureResult:
        """Classify the current pose."""
        if angles is None or not _frame_is_finite(angles):
            return self._handle_missing()

        self._missing_frames = 0

        best: Optional[GestureResult] = None
        for gesture in self.gestures:
            confidence = self._match(gesture, angles, landmarks)
            if confidence is not None:
                label = gesture.label
                best = GestureResult(gesture.name, label, round(confidence, 3))
                break

        return self._debounce(best)

    # ------------------------------------------------------------------

    def _match(
        self,
        gesture: Gesture,
        angles: JointAngles,
        landmarks: Optional[HandLandmarks],
    ) -> Optional[float]:
        """Return the match confidence, or None if the pose does not match."""
        margins: List[float] = []

        for finger, required in gesture.fingers.items():
            if required == ANY:
                continue
            state = finger_state(angles, finger)
            if state != required:
                return None
            # Decisiveness comes from the PIP alone.
            #
            # Requiring the MCP to clear the same threshold contradicted
            # finger_state(), which ignores the MCP on purpose: the knuckle
            # moves over only 20-70 degrees and a closed finger often reads
            # above FLEXED_MAX.  With both joints demanded, a fist whose pinky
            # knuckle read 133 degrees matched no gesture at all -- the state
            # said "flexed" and the margin said "no", in the same call.
            pip = _axis(angles, f"{finger}_pip")
            if pip is None:
                return None
            margin = _state_margin(pip, required)
            if margin is None:
                return None
            margins.append(margin)

        for axis, required in gesture.thumb.items():
            if required == ANY:
                continue
            value = _axis(angles, axis)
            if value is None:
                return None
            margin = _thumb_margin(value, required)
            if margin is None:
                return None
            margins.append(margin)

        if gesture.tips_touching is not None:
            ratio = tip_touch_ratio(landmarks)
            if ratio is None or ratio > self.tip_touch_ratio:
                return None
            # Closer tips -> more decisive.
            margins.append(min(1.0, (self.tip_touch_ratio - ratio) / self.tip_touch_ratio))

        return min(margins) if margins else 0.0

    def _handle_missing(self) -> GestureResult:
        """Treat a dropped detection as a gap, not as "the pose ended".

        The gesture is held for a few frames so a single lost detection does
        not blank the readout.  Only a gap longer than ``missing_grace`` means
        the hand really left.
        """
        self._missing_frames += 1
        self._candidate = None
        self._candidate_frames = 0

        if self._missing_frames >= self.missing_grace:
            self._current = None
            self._current_label = ""
            return GestureResult()

        if self._current is not None:
            return GestureResult(self._current, self._current_label, 0.0)
        return GestureResult()

    def _debounce(self, result: Optional[GestureResult]) -> GestureResult:
        """Require the same pose for several frames before reporting it."""
        name = result.name if result else None

        if name is None:
            self._candidate = None
            self._candidate_frames = 0
            self._current = None
            return GestureResult()

        if name == self._current:
            self._candidate = None
            self._candidate_frames = 0
            return result

        if name == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = name
            self._candidate_frames = 1

        if self._candidate_frames >= self.hold_frames:
            self._current = name
            self._current_label = result.label if result else ""
            self._candidate = None
            self._candidate_frames = 0
            return result

        # The previous gesture stays in force while a different one is still
        # earning its frames.  Dropping it here blanked the readout for the
        # whole debounce window whenever one gesture turned straight into
        # another, and in gesture-driven mode let the hand fall back to the raw
        # tracking for those frames.  A pose that matches nothing still clears
        # the gesture at once (above).
        if self._current is not None:
            return GestureResult(self._current, self._current_label, 0.0)
        return GestureResult()
