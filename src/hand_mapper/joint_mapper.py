"""
Joint Angle Mapper.

Converts 21 MediaPipe hand landmarks into servo-compatible joint angles.

The mapping is measured in the 2D image plane (the x/y pixel coordinates of
the landmarks); the z coordinate MediaPipe returns is relative and noisy and
is deliberately not used for flexion angles:
  1. Three adjacent landmarks define a joint (A - B - C)
  2. The angle at B is the angle between the projected vectors BA and BC

This is a projection, not a true 3D joint angle: a finger angled toward or
away from the camera reads as more extended than it is.  A single camera
cannot observe out-of-plane flexion, so this limitation is inherent to the
sensing, not to this implementation.

For the bionic hand, we map to 12 servos:
  - Thumb: CMC (abduction), MCP (flexion), IP (flexion)     [3 servos]
  - Index: MCP (flexion), PIP (flexion)                      [2 servos]
  - Middle: MCP (flexion), PIP (flexion)                     [2 servos]
  - Ring: MCP (flexion), PIP (flexion)                       [2 servos]
  - Pinky: MCP (flexion), PIP (flexion)                      [2 servos]
  - Wrist: rotation (optional, derived from wrist orientation) [1 servo]

Total: 12 degrees of freedom (expandable)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..hand_tracking.mediapipe_tracker import HandLandmarks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JointAngles:
    """
    Computed joint angles for all servos in degrees [0, 180].

    Angle convention:
      - 180° = fully extended (straight finger)
      - 0°   = fully flexed (closed fist)
      - 90°  = neutral/mid position
    """

    thumb_cmc:  float = 140.0   # Thumb carpometacarpal (abduction plane)
    thumb_mcp:  float = 140.0   # Thumb metacarpophalangeal (flexion)
    thumb_ip:   float = 140.0   # Thumb interphalangeal (flexion)
    index_mcp:  float = 160.0   # Index MCP
    index_pip:  float = 170.0   # Index PIP
    middle_mcp: float = 160.0   # Middle MCP
    middle_pip: float = 170.0   # Middle PIP
    ring_mcp:   float = 160.0   # Ring MCP
    ring_pip:   float = 170.0   # Ring PIP
    pinky_mcp:  float = 160.0   # Pinky MCP
    pinky_pip:  float = 170.0   # Pinky PIP
    wrist_rot:  float = 90.0    # Wrist rotation (optional)

    # Tracking info
    confidence: float = 0.0
    timestamp: float = field(default_factory=lambda: 0.0)

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dict for motor control."""
        return {
            "thumb_cmc":  self.thumb_cmc,
            "thumb_mcp":  self.thumb_mcp,
            "thumb_ip":   self.thumb_ip,
            "index_mcp":  self.index_mcp,
            "index_pip":  self.index_pip,
            "middle_mcp": self.middle_mcp,
            "middle_pip": self.middle_pip,
            "ring_mcp":   self.ring_mcp,
            "ring_pip":   self.ring_pip,
            "pinky_mcp":  self.pinky_mcp,
            "pinky_pip":  self.pinky_pip,
            "wrist_rot":  self.wrist_rot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "JointAngles":
        return cls(**{k: data.get(k, 90.0) for k in cls.__dataclass_fields__
                     if k not in ("confidence", "timestamp")})

    @classmethod
    def default(cls) -> "JointAngles":
        """Return default (relaxed open hand) angles."""
        return cls()


# ---------------------------------------------------------------------------
# Joint Mapper
# ---------------------------------------------------------------------------

class JointMapper:
    """
    Maps MediaPipe hand landmarks to joint angles suitable for servo control.

    The mapping computes angles from sets of 3 adjacent landmarks using
    the dot-product formula:

        θ = arccos( (v1·v2) / (|v1| |v2|) )

    with additional processing for:
      - Thumb (different kinematic chain)
      - MCP abduction/adduction (angle deviation from finger plane)
      - Angle normalization and clamping

    Usage:
        mapper = JointMapper()
        angles = mapper.compute_angles(hand_landmarks)
        print(f"Index MCP: {angles.index_mcp:.1f}°")
    """

    # Servo angle limits for safety
    MIN_ANGLE = 0.0
    MAX_ANGLE = 180.0

    # Finger flexion: servo goes from 0° (closed) to 180° (open)
    # These calibration factors translate computed anatomical angles
    # into servo angles. Adjust per your mechanical design.
    ANGLE_OFFSETS = {
        "thumb_cmc":  30.0,  # Mechanical offset for thumb CMC
        "thumb_mcp":  20.0,
        "thumb_ip":   15.0,
        "index_mcp":   0.0,
        "index_pip":   0.0,
        "middle_mcp":  0.0,
        "middle_pip":  0.0,
        "ring_mcp":    0.0,
        "ring_pip":    0.0,
        "pinky_mcp":   0.0,
        "pinky_pip":   0.0,
    }

    def __init__(
        self,
        smoothing_enabled: bool = True,
        smoothing_alpha: float = 0.3,
        max_step: float = 5.0,
        forced_handedness: Optional[str] = None,
    ):
        """
        Args:
            smoothing_enabled: Apply exponential moving average smoothing.
            smoothing_alpha: EMA factor (0-1). Lower = smoother but more lag.
            max_step: Maximum angle change per frame for safety.
        """
        self.smoothing_enabled = smoothing_enabled
        self.smoothing_alpha = smoothing_alpha
        self.max_step = max_step
        # MediaPipe's left/right label comes from assuming a mirrored camera
        # feed, and it is wrong often enough that wrist_rot was pinned at 0 for
        # an entire measured capture.  When set, this overrides the label for
        # the axes whose sign depends on which hand this is.
        self.forced_handedness = forced_handedness

        # Previous angle state for smoothing
        self._prev_angles: Optional[JointAngles] = None

        # The most recent pose with clipping but no temporal filtering.  Kept
        # alongside the smoothed one because recognition and actuation want
        # different things from the same frame.
        self._instant_angles: Optional[JointAngles] = None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute_angles(self, landmarks: HandLandmarks) -> Optional[JointAngles]:
        """
        Compute joint angles from hand landmarks.

        Args:
            landmarks: Detected hand landmarks from HandTracker.

        Returns:
            JointAngles with all servo angles, or None if landmarks invalid.
        """
        if landmarks is None or not landmarks.is_valid():
            return None

        try:
            # 1. Compute raw anatomical angles
            raw = JointAngles()

            # Thumb CMC - angle between thumb metacarpal and index metacarpal
            raw.thumb_cmc = self._angle_between(
                landmarks.landmarks_px[0],   # Wrist
                landmarks.landmarks_px[1],   # Thumb CMC
                landmarks.landmarks_px[5],   # Index MCP
            )

            # Thumb MCP - angle at thumb MCP joint
            raw.thumb_mcp = self._angle_between(
                landmarks.landmarks_px[1],   # Thumb CMC
                landmarks.landmarks_px[2],   # Thumb MCP
                landmarks.landmarks_px[3],   # Thumb IP
            )

            # Thumb IP - angle at thumb IP joint
            raw.thumb_ip = self._angle_between(
                landmarks.landmarks_px[2],   # Thumb MCP
                landmarks.landmarks_px[3],   # Thumb IP
                landmarks.landmarks_px[4],   # Thumb TIP
            )

            # Index finger
            raw.index_mcp = self._finger_mcp_angle(landmarks, 5)
            raw.index_pip = self._finger_pip_angle(landmarks, 5)

            # Middle finger
            raw.middle_mcp = self._finger_mcp_angle(landmarks, 9)
            raw.middle_pip = self._finger_pip_angle(landmarks, 9)

            # Ring finger
            raw.ring_mcp = self._finger_mcp_angle(landmarks, 13)
            raw.ring_pip = self._finger_pip_angle(landmarks, 13)

            # Pinky finger
            raw.pinky_mcp = self._finger_mcp_angle(landmarks, 17)
            raw.pinky_pip = self._finger_pip_angle(landmarks, 17)

            # Wrist rotation (derived from wrist-IndexMCP-PinkyMCP plane)
            raw.wrist_rot = self._compute_wrist_rotation(landmarks)

            # 2. Apply calibration offsets
            calibrated = self._apply_calibration(raw)

            # 3. Apply angle constraints (smoothing, clamping)
            result = self._apply_constraints(calibrated, landmarks)

            # 4. Update smoothing state
            self._prev_angles = result

            return result

        except (IndexError, ValueError) as e:
            logger.warning("Angle computation failed: %s", e)
            return None

    @property
    def instant_angles(self) -> Optional[JointAngles]:
        """The most recent pose with no temporal smoothing applied.

        Recognition reads this.  The smoothed values are eased toward the
        target over many frames -- that is what makes the actuators move
        smoothly, but it is pure lag to a classifier, and feeding the
        classifier the eased pose added the whole ramp to the delay between
        making a gesture and the UI reporting it.
        """
        return self._instant_angles

    def reset_smoothing(self) -> None:
        """Reset the smoothing state (e.g., after hand swap)."""
        self._prev_angles = None
        logger.debug("Smoothing state reset")

    # ------------------------------------------------------------------
    # Angle computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_between(
        p_before: np.ndarray,
        p_joint: np.ndarray,
        p_after: np.ndarray,
    ) -> float:
        """
        Compute the angle at p_joint formed by p_before-p_joint-p_after.

        Uses the dot product of vectors (p_before→p_joint) and (p_after→p_joint).

        Returns:
            Angle in degrees [0, 180].
        """
        v1 = p_before[:2] - p_joint[:2]   # Vector pointing INTO the joint
        v2 = p_after[:2] - p_joint[:2]     # Vector pointing OUT of the joint

        # Normalize
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0  # Fully extended if vectors are degenerate

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        # Clamp for numerical stability
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        if not np.isfinite(angle_deg):
            raise ValueError("non-finite joint angle")

        # The dot product gives us the interior angle (0-180).
        # Lower angle = more flexed, higher = more extended.
        # This maps naturally: 180° = straight, 0° = fully bent.
        return float(angle_deg)

    def _finger_mcp_angle(self, landmarks: HandLandmarks, mcp_idx: int) -> float:
        """
        Compute MCP flexion angle for a finger.

        Uses the wrist → MCP → PIP chain.
        mcp_idx: index of the MCP landmark (5=index, 9=middle, 13=ring, 17=pinky).
        """
        wrist = landmarks.landmarks_px[0]
        mcp = landmarks.landmarks_px[mcp_idx]
        pip = landmarks.landmarks_px[mcp_idx + 1]

        angle = self._angle_between(wrist, mcp, pip)
        return angle

    def _finger_pip_angle(self, landmarks: HandLandmarks, mcp_idx: int) -> float:
        """
        Compute PIP flexion angle for a finger.

        Uses the MCP → PIP → DIP chain.
        """
        mcp = landmarks.landmarks_px[mcp_idx]
        pip = landmarks.landmarks_px[mcp_idx + 1]
        dip = landmarks.landmarks_px[mcp_idx + 2]

        angle = self._angle_between(mcp, pip, dip)
        return angle

    def _compute_wrist_rotation(self, landmarks: HandLandmarks) -> float:
        """
        Estimate wrist pronation/supination from the palm's on-screen roll.

        Measured relative to the hand's own long axis (wrist -> middle MCP),
        not to the image horizontal.  Using the image horizontal would report
        the hand's absolute tilt in frame, so simply moving the arm would look
        like wrist twist; the anatomical signal is the palm's rotation about
        the forearm.

        The signed angle is mirrored for a left hand so that the same physical
        motion produces the same value on either hand, and shifted so that 90
        is the neutral palm-flat pose.
        """
        handedness = self.forced_handedness or landmarks.handedness

        wrist = landmarks.landmarks_px[0][:2]
        middle_mcp = landmarks.landmarks_px[9][:2]
        index_mcp = landmarks.landmarks_px[5][:2]
        pinky_mcp = landmarks.landmarks_px[17][:2]

        forearm = middle_mcp - wrist          # long axis, pointing to the hand
        lateral = pinky_mcp - index_mcp       # across the palm

        if np.linalg.norm(forearm) < 1e-6 or np.linalg.norm(lateral) < 1e-6:
            return 90.0

        # Reference is the palm-flat direction: perpendicular to the long
        # axis, on the pinky side for a right hand.
        perp = np.array([-forearm[1], forearm[0]])
        if handedness == "Left":
            perp = -perp

        cos_angle = np.dot(lateral, perp) / (
            np.linalg.norm(lateral) * np.linalg.norm(perp)
        )
        # Cross product gives the sign of the rotation (z component of the
        # 2D cross product) so pronation and supination differ.
        sign = np.sign(lateral[0] * perp[1] - lateral[1] * perp[0]) or 1.0
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))) * sign

        if handedness == "Left":
            angle_deg = -angle_deg

        return float(np.clip(angle_deg + 90.0, 0.0, 180.0))

    # ------------------------------------------------------------------
    # Calibration & constraints
    # ------------------------------------------------------------------

    def _apply_calibration(self, raw: JointAngles) -> JointAngles:
        """Apply mechanical calibration offsets to raw anatomical angles."""
        calibrated = JointAngles()

        for field_name in JointAngles.__dataclass_fields__:
            if field_name in ("confidence", "timestamp"):
                continue
            offset = self.ANGLE_OFFSETS.get(field_name, 0.0)
            raw_value = getattr(raw, field_name)
            calibrated_value = raw_value + offset
            setattr(calibrated, field_name, calibrated_value)

        return calibrated

    def _apply_constraints(
        self, angles: JointAngles, landmarks: HandLandmarks
    ) -> JointAngles:
        """Apply safety clamping and temporal smoothing to angles."""
        result = JointAngles()
        instant = JointAngles()
        # The Tasks API exposes no landmark-presence score; the handedness
        # classifier's score is the only per-frame quality signal available.
        result.confidence = landmarks.handedness_confidence
        result.timestamp = landmarks.timestamp
        instant.confidence = result.confidence
        instant.timestamp = result.timestamp

        for field_name in JointAngles.__dataclass_fields__:
            if field_name in ("confidence", "timestamp"):
                continue

            new_value = float(getattr(angles, field_name))

            # Clamp to safe range
            new_value = np.clip(new_value, self.MIN_ANGLE, self.MAX_ANGLE)

            setattr(instant, field_name, new_value)

            # Smoothing
            if self.smoothing_enabled and self._prev_angles is not None:
                prev_value = float(getattr(self._prev_angles, field_name))

                # Exponential moving average first, then a rate limit on the
                # step it produced.
                #
                # The order matters.  Capping the target *before* the EMA
                # compounds the two filters: a 5 degree cap followed by a 0.3
                # blend moves only 5 * 0.3 = 1.5 degrees per frame, so a 100
                # degree pose change needed 67 frames -- 2.2 s at 30 fps.
                # Filtering first leaves the cap as the only rate limit, and
                # it then means what it says: degrees per frame.
                step = self.smoothing_alpha * (new_value - prev_value)
                if abs(step) > self.max_step:
                    step = float(np.sign(step)) * self.max_step
                new_value = prev_value + step

            setattr(result, field_name, new_value)

        self._instant_angles = instant
        return result
