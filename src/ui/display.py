"""
Hand Display / Visualization.

Provides real-time visual feedback windows:
  1. Camera view with hand landmarks overlaid
  2. Virtual bionic hand (articulated skeleton from joint angles)
  3. Compact joint angle readout
  4. System status indicators

Designed for debugging, calibration, and demonstration without hardware.
"""

import logging
from typing import Dict, Optional

import cv2
import numpy as np

from ..hand_tracking.mediapipe_tracker import HandLandmarks
from ..hand_mapper.joint_mapper import JointAngles
from .virtual_hand import VirtualHandRenderer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

COLORS = {
    "background": (30, 30, 30),
    "panel_bg": (38, 38, 38),
    "text": (240, 240, 240),
    "bar_full": (0, 220, 100),
    "bar_mid": (0, 190, 255),
    "bar_low": (255, 120, 60),
    "bar_bg": (55, 55, 55),
    "finger_colors": {
        "thumb":  (255, 130, 80),
        "index":  (80, 230, 80),
        "middle": (80, 180, 255),
        "ring":   (230, 230, 80),
        "pinky":  (230, 130, 230),
    },
    "status_ok": (0, 230, 0),
    "status_warn": (0, 230, 230),
    "status_error": (0, 0, 255),
    "divider": (70, 70, 70),
}


# ---------------------------------------------------------------------------
# Hand Display
# ---------------------------------------------------------------------------

class HandDisplay:
    """
    Real-time visualization for the bionic hand system.

    Layout:
      ┌─────────────────────────────────────────────────────┐
      │  Status Bar (FPS, hand info, mode)                  │
      ├────────────────────┬────────────────────────────────┤
      │                    │  Virtual Bionic Hand           │
      │  Camera View       │  (articulated skeleton)        │
      │  (640 x 480)       │                                │
      │  + landmarks       ├────────────────────────────────┤
      │                    │  Joint Angles (compact bars)   │
      └────────────────────┴────────────────────────────────┘
    """

    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    DASHBOARD_WIDTH = 420
    VIRTUAL_HAND_HEIGHT = 350

    def __init__(
        self,
        show_camera: bool = True,
        show_dashboard: bool = True,
        window_name: str = "Bionic Hand Control",
        fullscreen: bool = False,
    ):
        self.show_camera = show_camera
        self.show_dashboard = show_dashboard
        self.window_name = window_name
        self.fullscreen = fullscreen

        self._is_open = False
        self._frame: Optional[np.ndarray] = None
        self._landmarks: Optional[HandLandmarks] = None
        self._angles: Optional[JointAngles] = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._last_key: int = -1
        self._gesture_name: Optional[str] = None
        self._finger_states: Dict[str, str] = {}

        # Virtual hand renderer
        self._virtual_hand = VirtualHandRenderer(
            width=self.DASHBOARD_WIDTH,
            height=self.VIRTUAL_HAND_HEIGHT,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        flags = cv2.WINDOW_NORMAL
        if self.fullscreen:
            flags |= cv2.WINDOW_FULLSCREEN
        cv2.namedWindow(self.window_name, flags)
        self._is_open = True
        logger.info("Display opened: %s", self.window_name)

    def close(self) -> None:
        if self._is_open:
            cv2.destroyWindow(self.window_name)
            self._is_open = False
            logger.info("Display closed")

    @property
    def is_open(self) -> bool:
        return self._is_open and cv2.getWindowProperty(
            self.window_name, cv2.WND_PROP_VISIBLE
        ) >= 1

    @property
    def last_key(self) -> int:
        """Return the key captured by the most recent show() call."""
        return self._last_key


    def update(
        self,
        camera_frame: np.ndarray,
        landmarks: Optional[HandLandmarks],
        angles: Optional[JointAngles],
        fps: float = 0.0,
        gesture_name: Optional[str] = None,
        finger_states: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        self._landmarks = landmarks
        # Hold the last pose across a dropped detection.  MediaPipe loses the
        # hand for a frame at a time -- 16% of frames in a measured capture --
        # and rendering the rest pose for those frames made the demo hand snap
        # to neutral and back several times a second.  That reads as stutter,
        # not as "the hand left".
        if angles is not None:
            self._angles = angles
        self._fps = fps
        self._gesture_name = gesture_name
        self._finger_states = finger_states or {}
        self._frame_count += 1

        if self.show_camera and self.show_dashboard:
            self._frame = self._build_composite(camera_frame)
        elif self.show_camera:
            self._frame = self._build_camera_only(camera_frame)
        else:
            self._frame = self._build_dashboard_only()

        return self._frame

    def show(self, delay_ms: int = 1) -> bool:
        if self._frame is None or not self.is_open:
            return False
        cv2.imshow(self.window_name, self._frame)
        self._last_key = cv2.waitKey(delay_ms) & 0xFF
        if self._last_key == ord("q") or self._last_key == 27:
            return False
        return True

    # ------------------------------------------------------------------
    # Composite builder
    # ------------------------------------------------------------------

    def _build_composite(self, camera_frame: np.ndarray) -> np.ndarray:
        # Camera view
        cam = cv2.resize(camera_frame, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
        if self._landmarks is not None:
            cam = self._draw_landmarks_on_frame(cam)

        # Dashboard (virtual hand + angle bars)
        dashboard = self._build_dashboard_panel()

        # Make them same height
        target_h = max(cam.shape[0], dashboard.shape[0])
        cam = self._pad_to_height(cam, target_h)
        dashboard = self._pad_to_height(dashboard, target_h)

        # Stack horizontally
        composite = np.hstack([cam, dashboard])

        # Top status bar
        status = self._build_status_bar(composite.shape[1])
        composite = np.vstack([status, composite])

        return composite

    def _build_camera_only(self, camera_frame: np.ndarray) -> np.ndarray:
        cam = cv2.resize(camera_frame, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
        if self._landmarks is not None:
            cam = self._draw_landmarks_on_frame(cam)
        return cam

    def _build_dashboard_only(self) -> np.ndarray:
        return self._build_dashboard_panel()

    # ------------------------------------------------------------------
    # Dashboard panel (Virtual Hand + Angle Bars)
    # ------------------------------------------------------------------

    def _build_dashboard_panel(self) -> np.ndarray:
        """Build the right-side panel: gesture info + virtual hand + angle bars."""
        # Get angles dict
        angle_map = self._angles.to_dict() if self._angles else {}

        # --- 1. Gesture info strip ---
        gesture_img = self._build_gesture_strip()

        # --- 2. Virtual Hand ---
        virtual_img = self._virtual_hand.render(angle_map)

        # --- 3. Compact angle bars ---
        bars_img = self._build_angle_bars(angle_map)

        # Stack vertically
        panel = np.vstack([gesture_img, virtual_img, bars_img])
        return panel

    def _build_gesture_strip(self) -> np.ndarray:
        """Build a strip showing the current gesture and per-finger state."""
        panel_h = 72
        panel = np.full((panel_h, self.DASHBOARD_WIDTH, 3),
                        COLORS["panel_bg"], dtype=np.uint8)

        # Title
        cv2.putText(panel, "Gesture", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text"], 1)

        # Gesture name (big)
        if self._gesture_name:
            gesture_text = self._gesture_name
            color = COLORS["status_ok"]
        else:
            gesture_text = "(none)"
            color = COLORS["status_warn"]
        cv2.putText(panel, gesture_text, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Per-finger mini-indicators
        fingers = [
            ("T", "thumb",  COLORS["finger_colors"]["thumb"]),
            ("I", "index",  COLORS["finger_colors"]["index"]),
            ("M", "middle", COLORS["finger_colors"]["middle"]),
            ("R", "ring",   COLORS["finger_colors"]["ring"]),
            ("P", "pinky",  COLORS["finger_colors"]["pinky"]),
        ]
        x = 260
        for label, fkey, base_color in fingers:
            state = self._finger_states.get(fkey, "?")
            # Circle fill color: green=extended, orange=flexed, gray=unclear
            if state == "extended":
                fill = (0, 220, 80)
                state_label = "E"
            elif state == "flexed":
                fill = (0, 120, 255)
                state_label = "F"
            else:
                fill = (100, 100, 100)
                state_label = "?"
            cv2.circle(panel, (x, 38), 12, fill, -1)
            cv2.circle(panel, (x, 38), 12, (60, 60, 60), 1)
            cv2.putText(panel, label, (x - 5, 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            # State tag below
            cv2.putText(panel, state_label, (x - 5, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, base_color, 1)
            x += 32

        # Bottom divider
        cv2.line(panel, (0, panel_h - 1), (self.DASHBOARD_WIDTH, panel_h - 1),
                 COLORS["divider"], 1)

        return panel

    def _build_angle_bars(self, angle_map: Dict[str, float]) -> np.ndarray:
        """Build compact horizontal angle bars for each servo."""
        bar_h = 18
        bar_gap = 4
        bar_max_w = 140
        label_w = 110
        num_bars = 12
        padding = 10

        panel_w = self.DASHBOARD_WIDTH
        panel_h = num_bars * (bar_h + bar_gap) + bar_gap + 30

        panel = np.full((panel_h, panel_w, 3), COLORS["panel_bg"], dtype=np.uint8)

        # Title
        cv2.putText(panel, "Servo Angles", (padding, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text"], 1)

        # Joint order
        joints = [
            ("T-CMC", "thumb_cmc", "thumb"),
            ("T-MCP", "thumb_mcp", "thumb"),
            ("T-IP",  "thumb_ip",  "thumb"),
            ("I-MCP", "index_mcp", "index"),
            ("I-PIP", "index_pip", "index"),
            ("M-MCP", "middle_mcp", "middle"),
            ("M-PIP", "middle_pip", "middle"),
            ("R-MCP", "ring_mcp", "ring"),
            ("R-PIP", "ring_pip", "ring"),
            ("P-MCP", "pinky_mcp", "pinky"),
            ("P-PIP", "pinky_pip", "pinky"),
            ("Wrist", "wrist_rot", None),
        ]

        bar_start_x = padding + label_w
        y = 26

        for label, key, finger_group in joints:
            angle = angle_map.get(key, 90.0)

            # Finger color indicator dot
            if finger_group:
                dot_color = COLORS["finger_colors"].get(finger_group, (150, 150, 150))
                cv2.circle(panel, (padding, y + bar_h // 2), 4, dot_color, -1)

            # Label
            cv2.putText(panel, label, (padding + 10, y + bar_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLORS["text"], 1)

            # Bar
            bar_x = bar_start_x
            bar_y = y
            cv2.rectangle(panel, (bar_x, bar_y),
                          (bar_x + bar_max_w, bar_y + bar_h),
                          COLORS["bar_bg"], -1)

            fill_w = int((angle / 180.0) * bar_max_w)
            if angle > 140:
                c = COLORS["bar_full"]
            elif angle > 60:
                c = COLORS["bar_mid"]
            else:
                c = COLORS["bar_low"]

            cv2.rectangle(panel, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), c, -1)
            cv2.rectangle(panel, (bar_x, bar_y),
                          (bar_x + bar_max_w, bar_y + bar_h),
                          (80, 80, 80), 1)

            # Angle value
            cv2.putText(panel, f"{int(angle)}",
                        (bar_x + bar_max_w + 6, bar_y + bar_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["text"], 1)

            y += bar_h + bar_gap

        # Detection info at bottom
        if self._landmarks is not None:
            info = f"{self._landmarks.handedness} hand  {self._landmarks.handedness_confidence:.0%}"
            cv2.putText(panel, info, (padding, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["status_ok"], 1)

        return panel

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self, total_width: int) -> np.ndarray:
        bar = np.full((30, total_width, 3), COLORS["background"], dtype=np.uint8)

        cv2.putText(bar, f"FPS: {self._fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

        if self._landmarks is not None:
            cv2.putText(bar,
                        f"Hand: {self._landmarks.handedness}  "
                        f"score: {self._landmarks.handedness_confidence:.2f}",
                        (110, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["status_ok"], 1)
        else:
            cv2.putText(bar, "No hand detected - show your hand to camera",
                        (110, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["status_warn"], 1)

        # Mode + quit hint
        hints_x = total_width - 240
        cv2.putText(bar, "Simulation Mode  |  Q=Quit  P=Pause  D=Reset",
                    (hints_x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        return bar

    # ------------------------------------------------------------------
    # Landmark drawing
    # ------------------------------------------------------------------

    def _draw_landmarks_on_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._landmarks is None:
            return frame

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        for i, j in connections:
            pt1 = tuple(self._landmarks.landmarks_px[i][:2].astype(int))
            pt2 = tuple(self._landmarks.landmarks_px[j][:2].astype(int))
            cv2.line(frame, pt1, pt2, (220, 220, 220), 1)

        for i, (px, py, _) in enumerate(self._landmarks.landmarks_px):
            cx, cy = int(px), int(py)
            if i in (4, 8, 12, 16, 20):
                color, radius = (0, 255, 255), 6
            elif i == 0:
                color, radius = (255, 0, 0), 5
            else:
                color, radius = (0, 255, 0), 4

            # Color by finger
            if i in (4,):
                color = COLORS["finger_colors"]["thumb"]
            elif i in (5, 6, 7, 8):
                color = COLORS["finger_colors"]["index"]
            elif i in (9, 10, 11, 12):
                color = COLORS["finger_colors"]["middle"]
            elif i in (13, 14, 15, 16):
                color = COLORS["finger_colors"]["ring"]
            elif i in (17, 18, 19, 20):
                color = COLORS["finger_colors"]["pinky"]
            elif i == 0:
                color = (200, 200, 200)

            cv2.circle(frame, (cx, cy), radius, color, -1)

        return frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        """Pad image to target height with background color."""
        h, w = img.shape[:2]
        if h >= target_h:
            return img
        pad = target_h - h
        padding = np.full((pad, w, 3), COLORS["background"], dtype=np.uint8)
        return np.vstack([img, padding])
