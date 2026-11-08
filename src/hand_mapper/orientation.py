"""
Hand orientation: which way the palm is facing.

Answers "is the operator showing me the palm, the back of the hand, or the edge
of the hand?" from the landmark geometry alone.

Why not use MediaPipe's z
-------------------------
MediaPipe does report a z per landmark, but it is a relative depth with a lot
of noise, and it is normalised in a way that makes absolute thresholds
meaningless.  The rest of this project already ignores z for angles
(see joint_mapper), and the same reasoning applies here.

The method: signed area of the palm triangle
--------------------------------------------
Three landmarks lie on the palm: the wrist (0), the index knuckle (5) and the
pinky knuckle (17).  In 3D they sit on the palm's plane; projected to 2D they
form a triangle whose shape encodes the viewing angle:

* Palm facing the camera  -> the triangle is seen face-on, maximum area.
* Palm edge-on            -> the triangle collapses toward a line, area -> 0.

Normalising by the two edge lengths makes it scale-invariant::

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    signed = (v1 x v2) / (|v1| * |v2|)          # = sin(angle between them)

``signed`` lands in [-1, +1] regardless of how far away the hand is:

* ``|signed|`` near 1  -> palm broadside to the camera
* ``|signed|`` near 0  -> hand edge-on
* the SIGN             -> which broadside (palm or back), and it flips with
                          handedness

The edge-on blind spot
----------------------
Edge-on is where this method is weakest, and that is geometry, not
implementation: when the palm is parallel to the view axis the triangle really
is a line, so the sign is determined by noise.  Two consequences shape the
design:

* "side" is a BAND, not a threshold, and it uses hysteresis so a hand hovering
  at the boundary does not flicker between states.
* the result carries a confidence, and low-confidence readings are reported as
  low confidence rather than dressed up as an answer.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..hand_tracking.mediapipe_tracker import HandLandmarks

logger = logging.getLogger(__name__)

# MediaPipe landmark indices used here.
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17

FACING_FRONT = "front"      # palm toward the camera
FACING_BACK = "back"        # back of the hand toward the camera
FACING_SIDE = "side"        # edge-on
FACING_UNKNOWN = "unknown"  # not enough information

# Hysteresis thresholds on |signed area|.
#
# A plain single threshold flickers: a hand held near the boundary crosses it
# every frame.  Entering "side" needs a clearly edge-on reading; leaving it
# needs a clearly broadside one; the gap between them is dead-band.
ENTER_SIDE = 0.30
EXIT_SIDE = 0.50

# Below this the reading is reported but flagged as low confidence.
LOW_CONFIDENCE = 0.45


@dataclass
class HandOrientation:
    """Result of one orientation estimate."""

    facing: str = FACING_UNKNOWN
    #: 0..1.  Near zero when the hand is edge-on, where the method is weakest.
    confidence: float = 0.0
    #: The normalised signed area, exposed for calibration and logging.
    signed_area: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.facing} ({self.confidence:.2f})"


def palm_signed_area(landmarks: HandLandmarks) -> Optional[float]:
    """Return the normalised signed area of the palm triangle, or None.

    ``None`` means the geometry was unusable -- degenerate points or a
    non-finite value -- which is different from a genuine edge-on reading of 0.
    """
    if landmarks is None:
        return None
    points = getattr(landmarks, "landmarks_px", None)
    if points is None:
        return None
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] <= PINKY_MCP or points.shape[1] < 2:
        return None

    wrist = points[WRIST][:2]
    index = points[INDEX_MCP][:2]
    pinky = points[PINKY_MCP][:2]
    if not (np.all(np.isfinite(wrist)) and np.all(np.isfinite(index))
            and np.all(np.isfinite(pinky))):
        return None

    v1 = index - wrist
    v2 = pinky - wrist
    len1 = float(np.linalg.norm(v1))
    len2 = float(np.linalg.norm(v2))
    if len1 < 1e-6 or len2 < 1e-6:
        return None

    # 2D cross product (the z component of v1 x v2).
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    return cross / (len1 * len2)


class OrientationDetector:
    """Classify hand orientation across frames, with hysteresis.

    Stateful on purpose: the hysteresis that stops the reading flickering at
    the edge-on boundary needs to remember which state it is currently in.

    Usage::

        detector = OrientationDetector()
        result = detector.update(landmarks)
        result.facing      # "front" | "back" | "side" | "unknown"
    """

    def __init__(
        self,
        *,
        enter_side: float = ENTER_SIDE,
        exit_side: float = EXIT_SIDE,
        invert_front_back: bool = False,
        forced_handedness: Optional[str] = None,
    ):
        self.enter_side = float(enter_side)
        self.exit_side = float(exit_side)
        #: Flip if calibration shows front and back are swapped.  This depends
        #: on whether the camera image is mirrored, so it must be settled once
        #: against a real camera rather than assumed.
        self.invert_front_back = bool(invert_front_back)
        #: Override the handedness the classifier reports.  Front/back is
        #: decided from the sign of the palm area, and that sign flips with the
        #: hand -- so a misclassified left/right silently inverts the answer.
        #: Pinning it ("Right") removes that failure mode for a single-user
        #: demo; leave None to trust MediaPipe.
        self.forced_handedness = forced_handedness
        self._in_side = False

    def reset(self) -> None:
        """Forget the hysteresis state (e.g. after a hand swap)."""
        self._in_side = False

    def update(self, landmarks: Optional[HandLandmarks]) -> HandOrientation:
        """Classify one frame."""
        if landmarks is None:
            return HandOrientation()

        signed = palm_signed_area(landmarks)
        if signed is None:
            return HandOrientation()

        magnitude = abs(signed)

        # Schmitt trigger: a different threshold to stay in "side" than to
        # enter it.  Without the gap a hand at the boundary alternates.
        if self._in_side:
            self._in_side = magnitude < self.exit_side
        else:
            self._in_side = magnitude < self.enter_side

        facing = (
            FACING_SIDE if self._in_side
            else self._classify_broadside(landmarks, signed)
        )
        return HandOrientation(
            facing,
            round(self._confidence(magnitude), 3),
            round(signed, 4),
        )

    def _confidence(self, magnitude: float) -> float:
        """How far the reading sits from the decision boundary, normalised.

        Confidence means "how sure is the classification", so it is HIGH at both
        extremes and LOW at the boundary:

        * a palm area near zero is confidently edge-on -- the geometry is
          unambiguous about the hand being side-on, even though it says nothing
          about which way the palm would face if it turned;
        * a large palm area is confidently broadside;
        * a reading sitting on the threshold is the uncertain case, because
          that is where "side" and "front/back" are genuinely hard to tell
          apart, and it is also where noise moves the answer.

        Framing this as "edge-on is low confidence" would be wrong: it is the
        boundary, not the edge-on pose, that is unreliable.
        """
        distance = abs(magnitude - self.enter_side)
        return min(1.0, distance / max(self.enter_side, 1e-6))

    def _classify_broadside(self, landmarks: HandLandmarks, signed: float) -> str:
        """Decide palm vs back for a broadside reading.

        The sign of the palm triangle's area flips when the hand turns over, and
        which sign means "palm" depends on the handedness MediaPipe reported.
        """
        handed = self.forced_handedness or str(
            getattr(landmarks, "handedness", "Right") or "Right"
        )
        # For a right hand, a positive area is the back of the hand toward the
        # camera.  A left hand is the mirror image, so the sense flips.
        positive_is_back = handed != "Left"
        if self.invert_front_back:
            positive_is_back = not positive_is_back
        if signed > 0:
            return FACING_BACK if positive_is_back else FACING_FRONT
        return FACING_FRONT if positive_is_back else FACING_BACK


def orientation_is_confident(orientation: HandOrientation) -> bool:
    """Whether a reading is solid enough to act on.

    Callers that drive hardware should gate on this: acting on a low-confidence
    edge-on reading means acting on noise.
    """
    return (
        orientation.facing != FACING_UNKNOWN
        and orientation.confidence >= LOW_CONFIDENCE
    )
