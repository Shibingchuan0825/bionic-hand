"""
MediaPipe Hand Tracker.

Provides real-time hand landmark detection using Google's MediaPipe Hands solution.
Detects 21 3D landmarks per hand with high accuracy.

Uses the modern MediaPipe Tasks API (0.10.x+).

Landmark reference (MediaPipe hand landmark model):
    0:  WRIST
    1:  THUMB_CMC
    2:  THUMB_MCP
    3:  THUMB_IP
    4:  THUMB_TIP
    5:  INDEX_FINGER_MCP
    6:  INDEX_FINGER_PIP
    7:  INDEX_FINGER_DIP
    8:  INDEX_FINGER_TIP
    9:  MIDDLE_FINGER_MCP
    10: MIDDLE_FINGER_PIP
    11: MIDDLE_FINGER_DIP
    12: MIDDLE_FINGER_TIP
    13: RING_FINGER_MCP
    14: RING_FINGER_PIP
    15: RING_FINGER_DIP
    16: RING_FINGER_TIP
    17: PINKY_MCP
    18: PINKY_PIP
    19: PINKY_DIP
    20: PINKY_TIP
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default model path (downloaded automatically)
_DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "hand_landmarker.task"

# Lazy imports for mediapipe (allows testing without it installed)
_mp = None
_vision = None
_Image = None
_ImageFormat = None
_BaseOptions = None
_HandLandmarker = None
_HandLandmarkerOptions = None
_HandLandmarksConnections = None


def _ensure_mediapipe():
    """Lazy-load mediapipe modules."""
    global _mp, _vision, _Image, _ImageFormat, _BaseOptions
    global _HandLandmarker, _HandLandmarkerOptions, _HandLandmarksConnections

    if _mp is not None:
        return

    import mediapipe as _mp_module
    from mediapipe.tasks import python as _python_module
    from mediapipe.tasks.python import vision as _vision_module
    from mediapipe import Image as _Image_cls
    from mediapipe import ImageFormat as _ImageFormat_cls
    from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions_cls

    _mp = _mp_module
    _vision = _vision_module
    _Image = _Image_cls
    _ImageFormat = _ImageFormat_cls
    _BaseOptions = _BaseOptions_cls
    _HandLandmarker = _vision_module.HandLandmarker
    _HandLandmarkerOptions = _vision_module.HandLandmarkerOptions
    _HandLandmarksConnections = _vision_module.HandLandmarksConnections

    # Monkey-patch: ensure connection tuples are hashable
    # HandLandmarksConnections is a list of tuples of ints which is fine


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HandLandmarks:
    """
    Processed hand landmarks from a single detection frame.
    Contains normalized (0-1 image coords) and pixel-space coordinates.
    """

    landmarks_px: np.ndarray   # shape: (21, 3) — pixel coordinates
    landmarks_norm: np.ndarray # shape: (21, 3) — normalized [0,1] coordinates
    handedness: str = "Right"
    # The Tasks API exposes exactly one per-hand score: the handedness
    # classifier's.  There is no separate landmark-presence/detection score, so
    # this single value is the only frame-quality signal available.  It is
    # named for what it actually is — do not read it as "detection confidence".
    handedness_confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    WRIST = 0
    THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
    INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks_px[self.WRIST]

    def get_finger_landmarks(self, finger_name: str) -> np.ndarray:
        ranges = {
            "thumb":  (1, 5), "index":  (5, 9), "middle": (9, 13),
            "ring":   (13, 17), "pinky":  (17, 21),
        }
        start, end = ranges[finger_name]
        return self.landmarks_px[start:end]

    def is_valid(self) -> bool:
        """Return whether this detection is safe for downstream geometry code."""
        try:
            pixels = np.asarray(self.landmarks_px)
            normalized = np.asarray(self.landmarks_norm)
            confidence = float(self.handedness_confidence)
        except (TypeError, ValueError):
            return False

        if pixels.shape != (21, 3) or normalized.shape != (21, 3):
            return False
        if not np.isfinite(confidence) or confidence <= 0.5:
            return False
        if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(normalized)):
            return False

        # A frame in which every landmark collapses to one pixel cannot provide
        # meaningful joint geometry, even though all values are finite.
        return bool(np.any(np.ptp(pixels[:, :2], axis=0) > 1e-6))


# ---------------------------------------------------------------------------
# Hand Tracker (MediaPipe Tasks API)
# ---------------------------------------------------------------------------

class HandTracker:
    """
    Real-time hand tracker using MediaPipe HandLandmarker (Tasks API 0.10.x).

    Usage:
        tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.7)
        tracker.start()
        while True:
            frame = camera.read()
            result = tracker.process(frame)
            if result:
                print(f"Detected {result.handedness} hand")
    """

    # Hand landmark connections (loaded lazily)
    _connections_cache = None

    @classmethod
    def _get_connections(cls):
        if cls._connections_cache is None:
            _ensure_mediapipe()
            cls._connections_cache = list(_HandLandmarksConnections)
        return cls._connections_cache

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ):
        # True  -> RunningMode.IMAGE: every frame is detected independently
        #          (use for still images / offline batches).
        # False -> RunningMode.VIDEO: landmarks are tracked across frames using
        #          monotonically increasing timestamps (use for a live camera).
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = model_path or str(_DEFAULT_MODEL_PATH)

        self._detector = None
        self._frame_count = 0
        self._total_process_time = 0.0
        self._fps = 0.0
        self._last_fps_update = time.time()
        self._last_timestamp_ms = -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize and start the MediaPipe HandLandmarker."""
        _ensure_mediapipe()

        # Download model if needed
        if not os.path.exists(self.model_path):
            self._download_model(self.model_path)

        running_mode = (
            _vision.RunningMode.IMAGE if self.static_image_mode
            else _vision.RunningMode.VIDEO
        )

        logger.info(
            "Starting HandTracker (model=%s, mode=%s, max_hands=%d, min_conf=%.2f)",
            self.model_path, running_mode.name, self.max_num_hands,
            self.min_detection_confidence,
        )

        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=self.model_path),
            running_mode=running_mode,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._detector = _HandLandmarker.create_from_options(options)
        self._frame_count = 0
        self._total_process_time = 0.0
        self._last_timestamp_ms = -1

        # Warmup: the first call triggers XNNPACK JIT compilation which can
        # take 30-60s on some CPUs. We do a dummy call here so subsequent
        # real-time calls are fast (~10ms).
        logger.info("Warming up model (first call may take 30-60s on some CPUs)...")
        warmup_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_warmup = _Image(image_format=_ImageFormat.SRGB, data=warmup_img)
        if self.static_image_mode:
            _ = self._detector.detect(mp_warmup)
        else:
            _ = self._detector.detect_for_video(mp_warmup, 0)
        logger.info("Model warmup complete, ready for real-time tracking")

    def stop(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            logger.info("HandTracker stopped")

    @property
    def is_running(self) -> bool:
        return self._detector is not None

    @property
    def fps(self) -> float:
        return self._fps

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        """
        Process a single video frame and detect hand landmarks.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            HandLandmarks if a hand is detected, None otherwise.
        """
        if self._detector is None:
            raise RuntimeError("HandTracker not started. Call start() first.")

        start_time = time.perf_counter()

        # MediaPipe Tasks API expects RGB Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = _Image(image_format=_ImageFormat.SRGB, data=frame_rgb)

        # Run detection.  VIDEO mode requires strictly increasing timestamps,
        # so clamp against the previous frame's value.
        if self.static_image_mode:
            result = self._detector.detect(mp_image)
        else:
            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
            result = self._detector.detect_for_video(mp_image, timestamp_ms)

        # Track performance
        elapsed = time.perf_counter() - start_time
        self._total_process_time += elapsed
        self._frame_count += 1
        self._update_fps()

        # Parse results
        if not result.hand_landmarks:
            return None

        # Take the first (most confident) hand
        hand_lm = result.hand_landmarks[0]
        handedness_list = result.handedness[0]

        # Convert to numpy
        h, w = frame.shape[:2]
        landmarks_norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm],
            dtype=np.float32,
        )
        landmarks_px = landmarks_norm.copy()
        landmarks_px[:, 0] *= w
        landmarks_px[:, 1] *= h

        # Handedness
        hand_label = handedness_list[0].category_name  # "Left" or "Right"
        hand_score = handedness_list[0].score

        logger.debug(
            "Frame %d: %s hand (conf=%.3f, time=%.1fms)",
            self._frame_count, hand_label, hand_score, elapsed * 1000,
        )

        return HandLandmarks(
            landmarks_px=landmarks_px,
            landmarks_norm=landmarks_norm,
            handedness=hand_label,
            handedness_confidence=hand_score,
            timestamp=time.time(),
        )

    def process_batch(self, frames: List[np.ndarray]) -> List[Optional[HandLandmarks]]:
        return [self.process(f) for f in frames]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: HandLandmarks,
        draw_connections: bool = True,
        draw_labels: bool = False,
    ) -> np.ndarray:
        """Draw hand landmarks and connections on the frame."""
        if landmarks is None:
            return frame

        for i, (px, py, _) in enumerate(landmarks.landmarks_px):
            cx, cy = int(px), int(py)
            if i in (4, 8, 12, 16, 20):
                color, radius = (0, 255, 255), 6   # Yellow fingertip
            elif i == 0:
                color, radius = (255, 0, 0), 5     # Blue wrist
            else:
                color, radius = (0, 255, 0), 4     # Green joint

            cv2.circle(frame, (cx, cy), radius, color, -1)
            if draw_labels:
                cv2.putText(frame, str(i), (cx + 8, cy + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if draw_connections:
            for idx1, idx2 in self._get_connections():
                pt1 = tuple(landmarks.landmarks_px[idx1][:2].astype(int))
                pt2 = tuple(landmarks.landmarks_px[idx2][:2].astype(int))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

        return frame

    def draw_info_overlay(self, frame: np.ndarray, landmarks: HandLandmarks) -> np.ndarray:
        y_offset = 30
        texts = [
            f"Hand: {landmarks.handedness} ({landmarks.handedness_confidence:.1%})",
            f"FPS: {self._fps:.1f}",
        ]
        for text in texts:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        return frame

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        now = time.time()
        elapsed = now - self._last_fps_update
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._total_process_time = 0.0
            self._last_fps_update = now

    @staticmethod
    def _download_model(target_path: Optional[str] = None) -> str:
        """Download the hand landmarker model to ``target_path``.

        The destination is resolved from the argument (defaulting to the
        bundled model path) so a custom ``model_path`` is honoured instead of
        being downloaded to the default location and then still missing.
        """
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        model_path = Path(target_path) if target_path else _DEFAULT_MODEL_PATH
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading hand landmarker model...")
        urllib.request.urlretrieve(url, str(model_path))
        logger.info("Model saved to %s", model_path)
        return str(model_path)
