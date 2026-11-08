"""
Dynamic (motion) gestures: what the hand is *doing*, not what shape it holds.

The static recogniser in ``gestures.py`` answers "is this a fist?".  This module
answers "is the hand moving in a circle?" -- a different question, and one that
cannot be answered from a single frame.

Design note: why not a trained model
------------------------------------
The well-known ``hand-gesture-recognition-using-mediapipe`` project ships a 6 KB
TFLite classifier for exactly these four motions.  It was evaluated and not
adopted: it needs TensorFlow (~500 MB) at runtime, ``tflite-runtime`` does not
install on Python 3.13, and the four classes are geometrically simple enough to
separate with a few measurements.  A rule set is also inspectable -- when a
gesture misreads you can see which number was on the wrong side of a threshold,
which is not true of a black-box model.

What is measured
----------------
Over a short window of frames the index fingertip traces a path.  Three
quantities separate the four classes:

* **path length** vs **net displacement** -- a circle travels far but ends near
  where it started; a straight move travels about as far as it ends up.
* **signed area** of the traced path (shoelace) -- its sign gives the direction
  of rotation.
* **speed** -- distinguishes standing still from any motion.

Everything is normalised by palm width, so the result does not depend on how far
the hand is from the camera.

The blind spot
--------------
A motion has to *finish* before it can be named, so these gestures are reported
one window late by construction.  That is inherent, not a tuning problem.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

import numpy as np

from ..hand_tracking.mediapipe_tracker import HandLandmarks

logger = logging.getLogger(__name__)

# MediaPipe landmark indices.
INDEX_TIP = 8
INDEX_MCP = 5
PINKY_MCP = 17

STOP = "stop"
CLOCKWISE = "clockwise"
COUNTER_CLOCKWISE = "counter_clockwise"
MOVE = "move"
UNKNOWN = "unknown"

# Frames held in the window.  At ~30 fps this is about half a second, long
# enough for one circling motion and short enough to feel responsive.
DEFAULT_WINDOW = 15

# A motion must cover at least this many palm widths to count as movement at
# all; below it the hand is considered still.
MIN_PATH_PALMS = 0.6

# Path length divided by net displacement.  A straight line scores ~1, a full
# circle scores well above it.
STRAIGHT_RATIO = 1.6

# Signed area, in palm-widths squared, above which the path counts as a loop
# rather than a wiggle.
MIN_LOOP_AREA = 0.25


@dataclass
class MotionResult:
    """Outcome of one dynamic-gesture classification."""

    name: str = UNKNOWN
    #: 0..1, how decisively the measurements separated the classes.
    confidence: float = 0.0
    path_palms: float = 0.0
    displacement_palms: float = 0.0
    signed_area: float = 0.0

    @property
    def matched(self) -> bool:
        return self.name != UNKNOWN


def _palm_width(points: np.ndarray) -> Optional[float]:
    """Distance between the index and pinky knuckles, in pixels."""
    a, b = points[INDEX_MCP][:2], points[PINKY_MCP][:2]
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    width = float(np.linalg.norm(b - a))
    return width if width > 1e-6 else None


def classify_path(
    points: Sequence[Tuple[float, float]],
    *,
    min_path_palms: float = MIN_PATH_PALMS,
    straight_ratio: float = STRAIGHT_RATIO,
    min_loop_area: float = MIN_LOOP_AREA,
) -> MotionResult:
    """Classify a traced path, already normalised to palm widths.

    Separated from landmark handling so the geometry can be tested with
    hand-built paths, which is the only way to check the circle cases exactly.
    """
    if len(points) < 4:
        return MotionResult()

    pts = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(pts)):
        return MotionResult()

    steps = np.diff(pts, axis=0)
    path = float(np.linalg.norm(steps, axis=1).sum())
    displacement = float(np.linalg.norm(pts[-1] - pts[0]))

    if path < min_path_palms:
        # Barely moved.  Reported as "stop" only when the window was actually
        # full, so a half-filled window is not mistaken for a still hand.
        return MotionResult(STOP, 1.0 - path / max(min_path_palms, 1e-6),
                            path, displacement, 0.0)

    # Shoelace formula about the path's own centroid, so the area does not
    # depend on where the hand sits in the image.
    centred = pts - pts.mean(axis=0)
    x, y = centred[:, 0], centred[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    ratio = path / max(displacement, 1e-6)
    if ratio >= straight_ratio:
        # Travelled much further than it ended up: a loop.
        if abs(area) < min_loop_area:
            return MotionResult(UNKNOWN, 0.0, path, displacement, area)
        name = CLOCKWISE if area < 0 else COUNTER_CLOCKWISE
        confidence = min(1.0, abs(area) / max(min_loop_area * 2.0, 1e-6))
        return MotionResult(name, confidence, path, displacement, area)

    # Roughly straight.  Confidence grows with how straight and how long.
    straightness = 1.0 - min(1.0, (ratio - 1.0) / max(straight_ratio - 1.0, 1e-6))
    length = min(1.0, displacement / max(min_path_palms * 2.0, 1e-6))
    return MotionResult(MOVE, min(straightness, length), path, displacement, area)


class MotionRecognizer:
    """Track the index fingertip across frames and name the motion.

    Stateful: the whole point is to remember the last half second.

    Usage::

        recognizer = MotionRecognizer()
        result = recognizer.update(landmarks)     # call once per frame
        if result.matched:
            ...
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        *,
        min_path_palms: float = MIN_PATH_PALMS,
        straight_ratio: float = STRAIGHT_RATIO,
        min_loop_area: float = MIN_LOOP_AREA,
    ):
        self.window = max(4, int(window))
        self.min_path_palms = float(min_path_palms)
        self.straight_ratio = float(straight_ratio)
        self.min_loop_area = float(min_loop_area)
        self._path: Deque[Tuple[float, float]] = deque(maxlen=self.window)

    def reset(self) -> None:
        """Forget the trail (e.g. when the hand leaves the frame)."""
        self._path.clear()

    def update(self, landmarks: Optional[HandLandmarks]) -> MotionResult:
        """Add one frame and classify the accumulated trail."""
        if landmarks is None:
            self.reset()
            return MotionResult()

        points = np.asarray(getattr(landmarks, "landmarks_px", None), dtype=float)
        if points.ndim != 2 or points.shape[0] <= PINKY_MCP or points.shape[1] < 2:
            return MotionResult()

        width = _palm_width(points)
        if width is None:
            return MotionResult()

        tip = points[INDEX_TIP][:2]
        if not np.all(np.isfinite(tip)):
            return MotionResult()

        # Normalise every sample to palm widths at the moment it was taken, so
        # moving the hand toward the camera does not look like motion.
        self._path.append((float(tip[0]) / width, float(tip[1]) / width))

        if len(self._path) < self.window:
            return MotionResult()

        return classify_path(
            list(self._path),
            min_path_palms=self.min_path_palms,
            straight_ratio=self.straight_ratio,
            min_loop_area=self.min_loop_area,
        )
