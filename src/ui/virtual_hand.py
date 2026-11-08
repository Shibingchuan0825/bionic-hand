"""
Virtual Bionic Hand Renderer.

Draws a dorsal-view (back of hand) hand that mirrors the bionic hand's joint
angles in real time.  The point of this view is to show the *mounting layout* of
the physical hand: the palm is the plate the fingers and the thumb bolt onto, so
the sketch draws that plate, the socket where each digit roots, and the web
between thumb and index that closes when the thumb adducts.

    * four fingers root along the palm's top edge and fan apart,
    * the thumb roots low on the lateral edge and swings up across the thenar,
    * the whole hand rotates about the wrist with wrist_rot.

Angles follow the servo convention: 0 degrees = fully flexed (closed),
180 degrees = fully extended (straight).

How flexion is drawn
--------------------
This is a dorsal view, so a finger curling toward the palm rotates about an
axis pointing roughly at the viewer: its true projection is *foreshortening*,
not a sideways swing.  The sketch therefore does two things at once, both of
them smooth functions of the command angle:

    * the segment shortens by ``cos`` of the fold, floored so a closed finger
      stays visible instead of collapsing to a point;
    * the segment tilts a little toward the palm's centre line, so the fold is
      readable rather than looking like the finger simply shrank.

An earlier version rotated each segment by the raw flexion angle instead.  That
is wrong in a way that only shows up numerically: subtracting 155 degrees from
an "up" direction lands near 180, so every finger in a fist swung out to the
side rather than curling in.  Interpolating toward a target direction has the
same failure, because "up" and "down" are antiparallel and the interpolated
vector passes through zero.  Foreshortening has neither problem.

No physics is simulated.  This is a kinematic sketch for operator feedback and
for checking the angle mapping before the mechanism exists.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# A hand reads as a hand when it is one material, not five.  The previous
# palette gave every finger its own colour, which looked like a stack of toy
# bricks: useful for telling fingers apart in a debug plot, wrong for a
# demonstration.  These are warm skin tones with a light from the upper left.
# NOTE: OpenCV colours are BGR, not RGB.  Writing a warm skin tone in RGB order
# here renders it cold blue -- which is exactly what happened the first time.
SKIN_LIGHT = (174, 194, 214)      # BGR for a light warm skin
SKIN_BASE = (146, 166, 188)
SKIN_SHADE = (110, 128, 150)
SKIN_EDGE = (78, 92, 108)

PALM_FILL = SKIN_BASE
PALM_SHADE = SKIN_SHADE
PALM_EDGE = SKIN_EDGE
THENAR_FILL = SKIN_LIGHT
SOCKET_FILL = SKIN_SHADE
BONE_COLOR = SKIN_EDGE
JOINT_COLOR = (150, 130, 114)
FINGERTIP_COLOR = (206, 188, 170)
BG_COLOR = (38, 38, 38)
TEXT_DIM = (150, 150, 150)
WRIST_FILL = (128, 146, 166)   # BGR, slightly darker than the palm

# All fingers share one material.  A slight tonal difference keeps them
# readable where they overlap without turning the hand into a colour chart.
FINGER_COLORS = {
    "thumb": (152, 172, 194),
    "index": (148, 168, 190),
    "middle": (144, 164, 186),
    "ring": (141, 160, 182),
    "pinky": (138, 156, 178),
}

# ---------------------------------------------------------------------------
# Geometry constants (design-space units, scaled to the canvas at draw time)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Segment proportions (RUKA v2 / human hand)
# ---------------------------------------------------------------------------
#
# The RATIOS below are the fixed, anatomical part.  They come from the RUKA v2
# hand model, which is itself built to human proportions:
#
#   four fingers, proximal : middle : distal  =  1 : 0.67 : 0.5
#   thumb,       proximal : distal            =  1 : 0.75
#   total length order                        =  middle > ring > index > pinky
#
# Only PALM_SCALE (and the palm outline it feeds) is meant to be re-tuned; the
# ratios should stay fixed.  That is the point of parameterising them: a model
# with hard-coded absolute sizes cannot be rescaled without redoing the linkage,
# which is why an off-the-shelf STL was not usable here.

# Proximal-phalanx ratio set for the four fingers (proximal, middle, distal).
FINGER_PHALANX_RATIOS = (1.00, 0.67, 0.50)
# The thumb has two phalanges, not three.
THUMB_PHALANX_RATIOS = (1.00, 0.75)

# Relative total finger length, longest = 1.  Order matters: middle > ring >
# index > pinky, which is what a real hand does.
FINGER_LENGTH_SCALE = {
    "index": 0.93,
    "middle": 1.00,
    "ring": 0.96,
    "pinky": 0.75,
}

# The thumb's visible length against the index's, phalanx for phalanx.
# A real thumb is about two thirds of the index (~57 mm against ~85 mm).
THUMB_TO_INDEX = 0.67

# The one size parameter: the middle finger's total phalanx length, in design
# units.  Everything else -- the other fingers, the thumb, the palm -- is
# derived from it and the ratios above, so there is a single number to tune.
MIDDLE_FINGER_LENGTH = 74.0

# Palm dimensions as multiples of the middle finger's TOTAL visible length
# (metacarpal + phalanges), not just the phalanges.  Measured from a real hand:
# palm height ~100 mm against a middle finger of ~95 mm, so the palm is a little
# taller than the finger is long.  Deriving the palm from the finger length
# means the two cannot drift apart when one is re-tuned -- which is exactly what
# happened when the fingers were lengthened and the palm kept its old size.
PALM_WIDTH_TO_FINGER = 0.95
PALM_HEIGHT_TO_FINGER = 1.05

# Per-digit thickness multiplier.  A thumb is noticeably chunkier than a little
# finger, and drawing every digit at one width makes the thumb read as a stray
# pole rather than as part of the hand.
WIDTH_SCALE = {
    "thumb": 1.55,
    "index": 1.10,
    "middle": 1.05,
    "ring": 1.00,
    "pinky": 0.85,
}

# Segment thickness as a fraction of the middle finger's LENGTH.  A real finger
# is about 0.19 of its own length thick at the proximal phalanx; an earlier
# value of ~0.08 drew them as wire, which is why the hand read as a stick
# figure rather than a hand.
FINGER_THICKNESS_RATIO = 0.19

# Thickness tapers toward the tip (a real finger does), as a fraction of the
# proximal thickness.
THICKNESS_TAPER = 0.72

# In-palm (metacarpal) segment per digit.  The four fingers have short ones
# because the palm body already spans that distance; the thumb's is longer
# because its metacarpal is buried in the thenar mass.
METACARPAL_LENGTH = {
    "thumb": 30.0,
    "index": 12.0,
    "middle": 13.0,
    "ring": 12.0,
    "pinky": 10.0,
}


def _derive_phalanx_lengths() -> Dict[str, tuple]:
    """Build the per-digit phalanx lengths from the ratios above.

    Derived rather than written out, so the proportions cannot drift apart from
    each other when one number is edited.
    """
    lengths: Dict[str, tuple] = {}
    for finger, scale in FINGER_LENGTH_SCALE.items():
        total = MIDDLE_FINGER_LENGTH * scale
        ratio_sum = sum(FINGER_PHALANX_RATIOS)
        lengths[finger] = tuple(total * r / ratio_sum for r in FINGER_PHALANX_RATIOS)

    index_total = MIDDLE_FINGER_LENGTH * FINGER_LENGTH_SCALE["index"]
    thumb_total = index_total * THUMB_TO_INDEX
    thumb_ratio_sum = sum(THUMB_PHALANX_RATIOS)
    lengths["thumb"] = tuple(thumb_total * r / thumb_ratio_sum
                             for r in THUMB_PHALANX_RATIOS)
    return lengths


PHALANX_LENGTHS = _derive_phalanx_lengths()

# Resting spread of the four fingers, in image degrees (0 = right, -90 = up).
FINGER_SPREAD = {
    "index": -97.0,
    "middle": -90.0,
    "ring": -83.0,
    "pinky": -76.0,
}

# Where each digit's metacarpal-phalangeal joint sits on the palm rim, in
# degrees around the palm centre (0 = right, -90 = up, 180 = left).
#
# The four knuckles fan across the TOP edge.  The angles are NOT evenly spaced:
# they are spaced in proportion to each finger's WIDTH, so that a thicker finger
# gets more room.  Evenly spaced roots put the middle and ring knuckles 13.5
# units apart while those fingers are 16-17 units thick, so they overlapped at
# the base.  Spacing by width is what a real hand does too.
#
# The thumb roots on the palm's lateral edge at roughly the palm's mid-height
# (about 180 degrees), NOT down at the bottom corner: rooted too low it could
# never reach the index tip in a pinch.
KNUCKLE_ANGLES = {
    "index": -109.9,
    "middle": -86.6,
    "ring": -64.6,
    "pinky": -45.3,
    "thumb": 213.0,
}

# Fraction of the palm radius at which each digit roots.  The thumb sits a
# little further out so the web has room to open.
KNUCKLE_RADIUS = {
    "index": 0.84,
    "middle": 0.86,
    "ring": 0.84,
    "pinky": 0.80,
    "thumb": 0.92,
}

# The angle range the renderer treats as "fully extended" .. "fully flexed".
#
# This is NOT 0-180, even though the protocol allows 0-180.  The mapper measures
# the interior angle at a joint, so a straight finger reads about 158 and a
# closed one about 68; it never emits the protocol extremes.  Calibrating the
# drawing against 0-180 therefore wasted most of the curve on angles that never
# occur, and a real hand closing moved the fingertip only ~12 units out of ~54
# -- which is exactly why the sketch looked like it was not following the hand.
#
# These bounds are measured from the mapper's actual output (see
# tests/test_virtual_hand.py::test_flex_curve_matches_the_mappers_real_range).
EXTENDED_ANGLE = 158.0
FLEXED_ANGLE = 68.0

# How far each joint bends in the drawing at full flexion, in degrees.
#
# These drawing angles control how much each phalanx rotates when the finger
# curls.  They are NOT the anatomical joint ranges, but they are close to them
# because the camera is looking at the hand from roughly the palm/dorsal side:
# fingers curl DOWN into the palm rather than into the camera, so the visible
# bend is the main effect and foreshortening is a secondary modifier.
# The old values (34/30/18) kept the hand looking like a bunch of straight
# sticks barely tilted; with realistic bend angles the fingers actually close.
FLEX_DEGREES = {
    "mcp": 62.0,
    "pip": 78.0,
    "dip": 40.0,
}

# Direction each finger points when fully extended, in image degrees
# (0 = right, -90 = straight up).
#
# These must track KNUCKLE_ANGLES: a finger rooted on the outside of the palm
# points outward, not straight up.  Leaving the directions clustered near -90
# while the roots fanned out made the hand look like a folding fan -- fingers
# radiating from a wide base but all aimed the same way.
FINGER_SPREAD = {
    "index": -103.0,
    "middle": -88.0,
    "ring": -74.0,
    "pinky": -58.0,
}


def _flex_sign(finger: str) -> float:
    """Which way a finger curls, in image-angle terms.

    Image angles run 0 = right, -90 = up, +/-180 = left, +90 = down.  On a
    dorsal view every finger curls the SAME rotational sense -- they all fold
    toward the palm, which means the direction angle increases for all four.

    Two earlier versions got this wrong in opposite ways, and both failures are
    worth keeping in mind:

    * Giving the fingers opposite signs (left ones one way, right ones the
      other) makes them converge on a single direction and CROSS over each
      other: the index ends up right of the ring finger.
    * Bending every finger by the same total angle still crosses them, because
      they start 45 degrees apart; the index needs a longer sweep than the
      pinky to reach the same place.

    Rotating all four the same way avoids both: they start spread and stay
    spread, each ending at its own angle inside the palm, which is what a real
    fist looks like.
    """
    return 1.0


# A closed finger keeps at least this fraction of its segment length.  Without
# a floor the foreshortening reaches zero and the digit vanishes.
# Minimum scale a fully-flexed phalanx foreshortens to.  With the bend angles
# above the fingers close properly; foreshortening just adds perspective depth
# rather than doing all the work of "looking curled".  0.15 means the distal
# segment still shows a sliver rather than vanishing entirely.
FORESHORTEN_FLOOR = 0.15

# The virtual DIP is not a servo axis; it follows the PIP as a single-tendon
# finger does.
DIP_COUPLING = 0.75

# Thumb CMC mapping: the raw thumb_cmc angle from MediaPipe does not span 0-180.
# In a measured 15 s open/fist capture it only moved 132-158 (26 degrees).
# Treating that 26-degree span as the full 0-180 command range would make the
# thumb swing wildly with even tiny jitter.  Instead we clamp the CMC input to
# the range that actually moves on a real hand, and map that to the full
# visual sweep.  The visual swing is what opens and closes the web.
THUMB_CMC_MIN = 120.0
THUMB_CMC_MAX = 170.0

# How far the thumb's two joints curl toward the palm at full flexion, as a
# fraction of the way toward the palm's centre.  A thumb folds less than a
# finger and in a different plane, so it gets its own constant rather than
# reusing FLEX_DEGREES.
THUMB_CURL = 0.5

# Thumb metacarpal direction, in image degrees, as a function of thumb_cmc.
# At 0 (flexed) the thumb points up and inward, across the palm toward the
# index; at 180 (extended) it is splayed out to the side.  That swing is what
# opens and closes the web, and it has to actually span the direction of the
# index tip or a pinch can never visually close -- the range was previously
# -114..-166, which never points to the right at all, so the thumb always
# pointed away from the fingers no matter what was commanded.
THUMB_DIR_AT_FLEXED = -58.0      # tucked in, across the palm toward the index
THUMB_DIR_AT_EXTENDED = -152.0   # splayed outward, web open


# The wrist_rot range the drawing is sized to fit.  90 is neutral.  The
# protocol allows 0-180, and the sketch is sized so that whole span stays on
# canvas -- a clipped fingertip is worse than a slightly smaller hand, because
# the operator cannot tell "the hand is at its limit" from "the drawing broke".
# Fitting a full 360 turn instead would shrink the hand to about half size.
FIT_WRIST_RANGE = (0.0, 180.0)


class VirtualHandRenderer:
    """Render a dorsal-view bionic hand from a joint-angle mapping.

    Usage:
        renderer = VirtualHandRenderer(width=420, height=350)
        image = renderer.render(angles)
    """

    def __init__(self, width: int = 420, height: int = 350):
        self.width = int(width)
        self.height = int(height)

        # The hand is laid out in a fixed design space and then scaled into
        # whatever canvas the caller asks for, so bone lengths never need
        # re-tuning when the dashboard changes size.
        self._design_w = 300.0
        self._design_h = 330.0

        # Palm size, derived from the finger length rather than fixed.  The two
        # must stay in proportion: lengthening the fingers while leaving the
        # palm at a hard-coded size is what made the hand look wrong.
        #
        # Measured against the middle finger's TOTAL length (metacarpal plus
        # phalanges), which is the length a person actually sees.
        visible_middle = (
            METACARPAL_LENGTH["middle"] + sum(PHALANX_LENGTHS["middle"])
        )
        palm_w = visible_middle * PALM_WIDTH_TO_FINGER
        palm_h = visible_middle * PALM_HEIGHT_TO_FINGER
        # _palm_radius is the half-height used throughout the drawing; the
        # half-width is derived from it in _palm_outline().
        self._palm_radius = palm_h / 2.0
        self._palm_half_width = palm_w / 2.0

        # Wrist pivot: low and centred.  The hand rotates about this point.
        self._wrist = (self._design_w / 2.0, self._design_h - 20.0)
        # Palm centre, measured up from the wrist along the palm axis.  Placed
        # so the palm's lower edge sits just above the wrist pivot.
        self._palm_center = (
            self._design_w / 2.0,
            self._wrist[1] - self._palm_radius - 8.0,
        )

        # Scale and offset are derived from the actual drawing, so the hand
        # fills the canvas no matter how the bone lengths above are tuned.
        self._scale, self._offset = self._compute_fit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, angles: Dict[str, float]) -> np.ndarray:
        """Render the hand.

        Args:
            angles: Joint name -> angle in degrees.  0 = flexed, 180 = extended.
                    Missing joints fall back to a relaxed open pose.

        Returns:
            BGR image as a numpy array.
        """
        canvas = np.full((self.height, self.width, 3), BG_COLOR, dtype=np.uint8)

        pose = self._solve(angles)

        # Wrist rotation swings every solved point about the wrist pivot.
        wrist_angle = float(angles.get("wrist_rot", 90.0))
        pose = self._apply_wrist_rotation(pose, wrist_angle)

        self._draw_title(canvas)
        self._draw_palm(canvas, pose, wrist_angle)

        # Drawing order depends on how far the thumb is folded.
        # When extended the thumb rides on the lateral edge of the palm and the
        # fingers overlap it; when curled it wraps across the front of the index
        # and middle fingers (that is how a real fist closes), so it must draw
        # last, on top.  Using the thumb IP fold fraction as the signal makes the
        # z-order crossfade happen where the curl happens rather than hard-
        # switching at one arbitrary angle.
        thumb_pip_angle = float(angles.get("thumb_ip", 170.0))
        thumb_folded = self._fold_fraction(thumb_pip_angle)
        draw_thumb_on_top = thumb_folded > 0.4

        if draw_thumb_on_top:
            for finger in ("index", "middle", "ring", "pinky"):
                self._draw_digit(canvas, pose, finger)
            self._draw_digit(canvas, pose, "thumb")
        else:
            self._draw_digit(canvas, pose, "thumb")
            for finger in ("index", "middle", "ring", "pinky"):
                self._draw_digit(canvas, pose, finger)
        self._draw_sockets(canvas, pose)
        self._draw_legend(canvas, wrist_angle)

        return canvas

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def _solve(self, angles: Dict[str, float]) -> Dict[str, List[Tuple[float, float]]]:
        """Compute the joint chain of every digit, in design space."""
        chains: Dict[str, List[Tuple[float, float]]] = {}
        for finger in ("index", "middle", "ring", "pinky"):
            chains[finger] = self._solve_finger(finger, angles)
        chains["thumb"] = self._solve_thumb(angles)
        return chains

    def _knuckle(self, finger: str) -> Tuple[float, float]:
        """Position of a digit's root socket on the palm rim."""
        cx, cy = self._palm_center
        radius = self._palm_radius * KNUCKLE_RADIUS[finger]
        ang = math.radians(KNUCKLE_ANGLES[finger])
        return (cx + radius * math.cos(ang), cy + radius * math.sin(ang))

    @staticmethod
    def _fold_fraction(servo_angle: float) -> float:
        """Map a measured joint angle onto 0 (extended) .. 1 (fully flexed).

        Calibrated against the mapper's real output range, not the protocol's
        0-180: see EXTENDED_ANGLE / FLEXED_ANGLE for why that matters.
        """
        span = EXTENDED_ANGLE - FLEXED_ANGLE
        return min(1.0, max(0.0, (EXTENDED_ANGLE - servo_angle) / span))

    def _solve_finger(
        self, finger: str, angles: Dict[str, float]
    ) -> List[Tuple[float, float]]:
        """Solve one of the four fingers: socket -> MCP -> PIP -> DIP -> TIP.

        Each phalanx rotates about its own joint, and the rotations ACCUMULATE
        down the chain -- that is what a finger does.

        The bend is measured from the finger's RESTING direction toward a target
        direction (straight down, into the palm), never by adding to the running
        direction.  Two failure modes came from getting this wrong:

        * Adding flexion to the running direction wraps a segment past
          horizontal, throwing the whole fist out sideways.
        * Bending every finger by the SAME total angle makes the outer fingers
          overshoot: the index starts 193 degrees from straight-down and the
          pinky only 148, so one shared angle sends the index across the ring
          finger.  Fingers crossed over each other in a fist.

        Aiming each finger at its own target keeps them converging without
        crossing, which is what a closing hand does.
        """
        base = self._knuckle(finger)
        prox_len, mid_len, dist_len = PHALANX_LENGTHS[finger]

        # The metacarpal runs outward from the socket along the finger's spread
        # direction.  It sits inside the palm, so it never folds.
        spread = FINGER_SPREAD[finger]
        mcp = self._step(base, spread, METACARPAL_LENGTH[finger])

        mcp_fold = self._fold_fraction(float(angles.get(f"{finger}_mcp", 160.0)))
        pip_fold = self._fold_fraction(float(angles.get(f"{finger}_pip", 170.0)))
        dip_fold = pip_fold * DIP_COUPLING

        sign = _flex_sign(finger)

        # Cumulative flexion at each joint, in drawing degrees.
        #
        # On a palm/dorsal view a finger curls DOWN into the palm, which is
        # primarily a rotation of each phalanx -- the visible bend is the main
        # effect.  Foreshortening adds a small amount of perspective shortening.
        # Earlier versions kept the bend tiny (34/30/18) and relied on foreshortening
        # to do all the work, which made closed fingers look like straight sticks
        # tilted slightly inward instead of a fist.
        mcp_bend = sign * mcp_fold * FLEX_DEGREES["mcp"]
        pip_bend = mcp_bend + sign * pip_fold * FLEX_DEGREES["pip"]
        dip_bend = pip_bend + sign * dip_fold * FLEX_DEGREES["dip"]

        # Foreshortening does the rest of the work: each segment shortens as its
        # joint folds, which is what actually happens when a finger curls toward
        # the camera.
        mcp_scale = 1.0 - (1.0 - FORESHORTEN_FLOOR) * mcp_fold
        pip_scale = 1.0 - (1.0 - FORESHORTEN_FLOOR) * pip_fold
        dip_scale = 1.0 - (1.0 - FORESHORTEN_FLOOR) * dip_fold

        pip = self._step(mcp, spread + mcp_bend, prox_len * mcp_scale)
        dip = self._step(pip, spread + pip_bend, mid_len * pip_scale)
        tip = self._step(dip, spread + dip_bend, dist_len * dip_scale)

        return [base, mcp, pip, dip, tip]

    def _solve_thumb(self, angles: Dict[str, float]) -> List[Tuple[float, float]]:
        """Solve the thumb: socket -> CMC -> MCP -> IP -> TIP.

        ``thumb_cmc`` swings the whole thumb between lying alongside the index
        (flexed) and standing away from the palm (extended).  That swing is what
        opens and closes the web, which is why the thumb is drawn from a low
        lateral socket rather than off the knuckle line.
        """
        base = self._knuckle("thumb")
        prox_len, dist_len = PHALANX_LENGTHS["thumb"]

        cmc_angle = float(angles.get("thumb_cmc", 140.0))
        mcp_fold = self._fold_fraction(float(angles.get("thumb_mcp", 140.0)))
        ip_fold = self._fold_fraction(float(angles.get("thumb_ip", 140.0)))

        # Map the CMC angle into the visual direction span.  MediaPipe's
        # thumb_cmc only moves about 20-30 degrees on a real hand; feeding that
        # raw angle straight into the visual swing makes the thumb jerk around
        # on tiny jitter -- a 2 degree noise spike becomes a 7 degree direction
        # change.  Clamping to the range that actually represents the open/close
        # travel gives a smooth, controllable web.
        t = (cmc_angle - THUMB_CMC_MIN) / max(1e-6, THUMB_CMC_MAX - THUMB_CMC_MIN)
        t = max(0.0, min(1.0, t))
        cmc_dir = THUMB_DIR_AT_FLEXED + (THUMB_DIR_AT_EXTENDED - THUMB_DIR_AT_FLEXED) * t

        # The metacarpal is buried in the thenar, so only the part that clears
        # the palm rim is drawn.  Drawing too little of it leaves the thumb
        # looking detached from the hand.
        mcp = self._step(base, cmc_dir, METACARPAL_LENGTH["thumb"] * 0.85)

        # The thumb curls in its own plane, rotating about the MCP axis.
        # In image coordinates (0=right, 90=down, -90=up), a splayed thumb points
        # up-left (-152) and a tucked thumb points up-right (-58) across the palm.
        # From either pose, flexion rotates the distal phalanx down toward the
        # palm: adding about 100 degrees clockwise from the metacarpal direction
        # puts the tip squarely across the front of the index finger, where it
        # belongs during a real fist.  The previous code aimed the tip at the
        # palm centre, which made it slide off to the right; a rotational
        # offset in the joint's own plane produces a clean arc that ends up in
        # the right place.
        inward_deg = 105.0
        mcp_dir = cmc_dir + inward_deg * mcp_fold
        ip = self._step(mcp, mcp_dir, prox_len)

        ip_inward = inward_deg * 0.6
        ip_dir = mcp_dir + ip_inward * ip_fold
        tip = self._step(ip, ip_dir, dist_len)

        return [base, mcp, ip, tip]

    @staticmethod
    def _blend(from_deg: float, to_deg: float, fraction: float) -> float:
        """Rotate ``from_deg`` toward ``to_deg`` by ``fraction`` of the way."""
        fraction = min(1.0, max(0.0, fraction))
        delta = (to_deg - from_deg + 180.0) % 360.0 - 180.0
        return from_deg + delta * fraction

    @staticmethod
    def _step(
        origin: Tuple[float, float], angle_deg: float, length: float
    ) -> Tuple[float, float]:
        rad = math.radians(angle_deg)
        return (origin[0] + length * math.cos(rad), origin[1] + length * math.sin(rad))

    # ------------------------------------------------------------------
    # Wrist rotation
    # ------------------------------------------------------------------

    def _rotate_point(
        self, point: Tuple[float, float], wrist_angle: float
    ) -> Tuple[float, float]:
        """Rotate one design-space point about the wrist pivot."""
        delta = math.radians(wrist_angle - 90.0)
        cos_d, sin_d = math.cos(delta), math.sin(delta)
        wx, wy = self._wrist
        dx, dy = point[0] - wx, point[1] - wy
        return (wx + dx * cos_d - dy * sin_d, wy + dx * sin_d + dy * cos_d)

    def _apply_wrist_rotation(
        self, pose: Dict[str, List[Tuple[float, float]]], wrist_angle: float
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Rotate every solved point about the wrist pivot.

        ``wrist_rot`` is a yaw about the forearm axis.  Seen from the back of
        the hand it reads as the whole hand tilting, so the sketch tilts too.
        """
        return {
            finger: [self._rotate_point(p, wrist_angle) for p in chain]
            for finger, chain in pose.items()
        }

    # ------------------------------------------------------------------
    # Canvas fitting
    # ------------------------------------------------------------------

    def _reference_angles(self) -> Dict[str, float]:
        """A relaxed open pose, used to size the drawing."""
        angles: Dict[str, float] = {"wrist_rot": 90.0}
        for finger in ("index", "middle", "ring", "pinky"):
            angles[f"{finger}_mcp"] = 180.0
            angles[f"{finger}_pip"] = 180.0
        angles.update({"thumb_cmc": 150.0, "thumb_mcp": 170.0, "thumb_ip": 170.0})
        return angles

    def _compute_fit(self) -> Tuple[float, Tuple[float, float]]:
        """Derive scale and offset so the drawing fills the canvas.

        The fit is measured over the *whole* wrist_rot range, not just the
        upright pose.  A hand that fits at 90 degrees can swing far outside the
        canvas at 140: the fingers are drawn from a pivot near the bottom, so
        rotating sweeps them through a wide arc.  Fitting only the upright pose
        silently clipped fingertips off-screen whenever the wrist turned.

        Measuring the actual geometry also beats hand-tuned constants: if a bone
        length changes, the fit follows automatically.
        """
        points = list(self._palm_outline())
        points.extend(self._wrist_quad())
        upright = self._solve(self._reference_angles())
        for chain in upright.values():
            points.extend(chain)

        # Rotate the same geometry through the wrist range and include every
        # pose.  The wrist pivot is rotation-invariant, so it is added once.
        low, high = FIT_WRIST_RANGE
        samples = [low + (high - low) * i / 12.0 for i in range(13)]
        for wrist_angle in samples:
            rotated = self._apply_wrist_rotation(upright, wrist_angle)
            for chain in rotated.values():
                points.extend(chain)
            # Palm and wrist stub rotate too.
            delta = math.radians(wrist_angle - 90.0)
            cos_d, sin_d = math.cos(delta), math.sin(delta)
            wx, wy = self._wrist
            for x, y in list(self._palm_outline()) + list(self._wrist_quad()):
                dx, dy = x - wx, y - wy
                points.append((wx + dx * cos_d - dy * sin_d,
                               wy + dx * sin_d + dy * cos_d))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        margin = 6.0
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        scale = min(
            (self.width - 2 * margin) / span_x,
            (self.height - 2 * margin) / span_y,
        )
        offset = (
            (self.width - span_x * scale) / 2.0 - min_x * scale,
            (self.height - span_y * scale) / 2.0 - min_y * scale,
        )
        return scale, offset

    def _to_canvas(self, point: Tuple[float, float]) -> Tuple[int, int]:
        x = point[0] * self._scale + self._offset[0]
        y = point[1] * self._scale + self._offset[1]
        return (int(round(x)), int(round(y)))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_title(self, canvas: np.ndarray) -> None:
        cv2.putText(canvas, "Virtual Bionic Hand", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 205, 205), 1)

    def _palm_outline(self) -> List[Tuple[float, float]]:
        """Sample the palm silhouette.

        A real palm is widest across the knuckles and tapers steadily down to
        the wrist -- a rounded trapezoid.  Two earlier attempts missed this: a
        plain ellipse read as a blob, and a "waist plus heel flare" profile read
        as a vase.  The taper here is monotonic, which is what a palm does.
        """
        cx, cy = self._palm_center
        a = self._palm_half_width             # half width at the knuckle line
        b = self._palm_radius                 # half height
        n = 2.6                               # corner roundness

        points: List[Tuple[float, float]] = []
        steps = 96
        for i in range(steps):
            t = 2.0 * math.pi * i / steps
            ct, st = math.cos(t), math.sin(t)
            x = a * math.copysign(abs(ct) ** (2.0 / n), ct)
            y = b * math.copysign(abs(st) ** (2.0 / n), st)

            # v runs 0 at the knuckle line (top) to 1 at the wrist (bottom).
            v = (y / b + 1.0) / 2.0
            # Monotonic taper: widest at the knuckles, narrowest at the wrist.
            width = 1.0 - 0.30 * v
            points.append((cx + x * width, cy + y))
        return points

    def _knuckle_bulges(self) -> List[Tuple[float, float, float]]:
        """Centre and radius of the bulge at each finger's knuckle.

        Drawn on top of the palm so the knuckle line reads as knuckles rather
        than a flat edge.
        """
        cx, cy = self._palm_center
        r = self._palm_radius
        bulges = []
        for finger in ("index", "middle", "ring", "pinky"):
            kx, ky = self._knuckle(finger)
            # Pull each bulge slightly inside the rim so it sits ON the palm.
            bx = cx + (kx - cx) * 0.90
            by = cy + (ky - cy) * 0.90
            bulges.append((bx, by, r * 0.19))
        return bulges

    def _wrist_quad(self) -> List[Tuple[float, float]]:
        """The wrist stub below the palm, so the pivot is visible."""
        cx, cy = self._palm_center
        wx, wy = self._wrist
        top = cy + self._palm_radius * 0.55
        half_top = self._palm_radius * 0.34
        half_bottom = self._palm_radius * 0.30
        return [
            (cx - half_top, top),
            (cx + half_top, top),
            (wx + half_bottom, wy),
            (wx - half_bottom, wy),
        ]

    def _draw_palm(
        self,
        canvas: np.ndarray,
        pose: Dict[str, List[Tuple[float, float]]],
        wrist_angle: float,
    ) -> None:
        """Draw the wrist stub, the palm plate, and the thenar mass.

        The palm and wrist stub rotate with the hand.  Drawing them unrotated
        while the fingers rotated left every finger rooted in mid-air beside a
        stationary palm -- the digits and the plate they bolt onto have to share
        one transform.
        """
        rotate = lambda p: self._rotate_point(p, wrist_angle)  # noqa: E731

        # Wrist first, so the palm overlaps it.
        cv2.fillPoly(
            canvas,
            [np.array([self._to_canvas(rotate(p)) for p in self._wrist_quad()],
                      dtype=np.int32)],
            WRIST_FILL,
        )

        outline = np.array(
            [self._to_canvas(rotate(p)) for p in self._palm_outline()], dtype=np.int32
        )
        cv2.fillPoly(canvas, [outline], PALM_FILL)

        # Thickness: shade the lower (wrist-side) half darker.  A flat fill has
        # no depth, which is most of why the palm read as a paper cut-out.
        cx, cy = self._palm_center
        r = self._palm_radius
        shade = np.array(
            [
                self._to_canvas(rotate((cx - r * 1.05, cy + r * 0.10))),
                self._to_canvas(rotate((cx + r * 1.05, cy + r * 0.10))),
                self._to_canvas(rotate((cx + r * 1.05, cy + r * 1.05))),
                self._to_canvas(rotate((cx - r * 1.05, cy + r * 1.05))),
            ],
            dtype=np.int32,
        )
        # Clip the shading to the palm silhouette so it cannot spill outside.
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [outline], 255)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [shade], PALM_SHADE)
        blended = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
        canvas[mask > 0] = blended[mask > 0]

        # Knuckle bulges along the top edge.
        for bx, by, br in self._knuckle_bulges():
            cv2.circle(canvas, self._to_canvas(rotate((bx, by))),
                       max(3, int(br * self._scale)), PALM_EDGE, -1)
            cv2.circle(canvas, self._to_canvas(rotate((bx, by))),
                       max(3, int(br * self._scale)), PALM_FILL, 1)

        # Outline last, so nothing overlaps the silhouette edge.
        cv2.polylines(canvas, [outline], True, PALM_EDGE, 2)

        # Thenar mass: the wedge between the thumb root and the palm.  This is
        # the region the web closes into, so it is drawn from the solved thumb
        # and index roots rather than at a fixed spot.
        thumb_root = pose["thumb"][0]
        index_root = pose["index"][0]
        thenar = np.array(
            [
                self._to_canvas(thumb_root),
                self._to_canvas(rotate((cx - r * 0.74, cy + r * 0.34))),
                self._to_canvas(rotate((cx - r * 0.10, cy + r * 0.62))),
                self._to_canvas(index_root),
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [thenar], THENAR_FILL)
        cv2.polylines(canvas, [thenar], True, PALM_EDGE, 1)

        # Palm creases.  Drawn as curved arcs, not straight lines: straight
        # segments read as scratches on a flat block, which is worse than no
        # creases at all.
        for start, ctrl, end, shade in (
            ((-0.42, 0.10), (-0.10, 0.34), (0.16, 0.58), 0),   # life line
            ((-0.34, -0.20), (0.02, 0.06), (0.34, 0.14), 0),   # head line
            ((0.20, -0.30), (0.28, -0.02), (0.26, 0.26), 1),   # heart line
        ):
            pts = []
            for k in range(13):
                u = k / 12.0
                # Quadratic Bezier.
                px = (1 - u) ** 2 * start[0] + 2 * (1 - u) * u * ctrl[0] + u ** 2 * end[0]
                py = (1 - u) ** 2 * start[1] + 2 * (1 - u) * u * ctrl[1] + u ** 2 * end[1]
                pts.append(self._to_canvas(rotate((cx + r * px, cy + r * py))))
            colour = PALM_SHADE if shade else PALM_EDGE
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False,
                          colour, 1, cv2.LINE_AA)

    def _draw_sockets(
        self, canvas: np.ndarray, pose: Dict[str, List[Tuple[float, float]]]
    ) -> None:
        """Soften the base of each digit.

        These were filled circles marking the physical mounting points, which is
        honest but reads as bolts screwed into a plate.  A hand does not show
        its fixings, so the joint is now a soft shadow where the finger meets
        the palm -- the same information, without the hardware look.
        """
        radius = max(2, int(3.0 * self._scale))
        for chain in pose.values():
            centre = self._to_canvas(chain[0])
            cv2.circle(canvas, centre, radius, SKIN_SHADE, -1, cv2.LINE_AA)

    def _digit_polygon(
        self, chain: List[Tuple[float, float]], finger: str
    ) -> np.ndarray:
        """Build one tapered outline around a digit's whole joint chain.

        Drawn as a single polygon rather than one capsule per bone.  Capsules
        per bone make a chain of beads: each joint shows a bulge where the two
        circles overlap, which reads as a string of pearls instead of a finger.
        """
        widths = self._segment_widths(len(chain) - 1, finger)
        # Half-width at each joint, in canvas pixels.
        half = [w / 2.0 for w in widths]
        half.append(half[-1] * THICKNESS_TAPER)

        pts = [self._to_canvas(p) for p in chain]
        left: List[Tuple[int, int]] = []
        right: List[Tuple[int, int]] = []

        for i, (x, y) in enumerate(pts):
            # Direction of the segment leaving this joint (or entering, at the
            # tip), used to offset the outline perpendicular to the finger.
            if i < len(pts) - 1:
                dx, dy = pts[i + 1][0] - x, pts[i + 1][1] - y
            else:
                dx, dy = x - pts[i - 1][0], y - pts[i - 1][1]
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length      # unit normal
            h = half[min(i, len(half) - 1)]
            left.append((int(round(x + nx * h)), int(round(y + ny * h))))
            right.append((int(round(x - nx * h)), int(round(y - ny * h))))

        return np.array(left + right[::-1], dtype=np.int32)

    def _draw_digit(
        self, canvas: np.ndarray, pose: Dict[str, List[Tuple[float, float]]], finger: str
    ) -> None:
        """Draw one digit as a shaded, tapered outline.

        Three passes give it roundness without a 3D pipeline: a filled body, a
        lighter band down the lit side, and a dark outline.  A flat fill with a
        coloured outline reads as a diagram; this reads as a finger.
        """
        chain = pose[finger]
        colour = FINGER_COLORS[finger]
        polygon = self._digit_polygon(chain, finger)

        cv2.fillPoly(canvas, [polygon], colour)

        # Lit side: a thinner polygon offset toward the upper-left, where the
        # light comes from.  Clipped to the digit so it cannot spill.
        highlight = self._offset_polygon(polygon, dx=-2, dy=-2, shrink=0.42)
        if highlight is not None:
            mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [highlight], SKIN_LIGHT)
            blended = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
            canvas[mask > 0] = blended[mask > 0]

        cv2.polylines(canvas, [polygon], True, SKIN_EDGE, 1, cv2.LINE_AA)

        # Knuckles are a crease ACROSS the finger, not a dot on it.  Round
        # markers at every joint read as rivets and were the main thing making
        # the drawing look like a mechanical diagram.
        canvas_pts = [self._to_canvas(p) for p in chain]
        widths = self._segment_widths(len(chain) - 1, finger)
        for i in range(1, len(canvas_pts) - 1):
            x, y = canvas_pts[i]
            px, py = canvas_pts[i - 1]
            dx, dy = x - px, y - py
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            half = max(1, widths[min(i, len(widths) - 1)] // 2 - 1)
            cv2.line(canvas,
                     (int(x + nx * half), int(y + ny * half)),
                     (int(x - nx * half), int(y - ny * half)),
                     SKIN_SHADE, 1, cv2.LINE_AA)

    @staticmethod
    def _offset_polygon(polygon: np.ndarray, dx: int, dy: int,
                        shrink: float) -> Optional[np.ndarray]:
        """A smaller copy of a polygon, shifted to fake a light direction."""
        if polygon is None or len(polygon) < 3:
            return None
        centre = polygon.mean(axis=0)
        shifted = (polygon.astype(float) - centre) * (1.0 - shrink) + centre
        shifted += np.array([dx, dy], dtype=float)
        return shifted.astype(np.int32)

    def _segment_widths(self, count: int, finger: str = "middle") -> List[int]:
        """Line widths from knuckle to tip, in canvas pixels.

        Thickness is derived from the middle finger's LENGTH (a real finger is
        about 0.19 of its own length thick at the proximal phalanx), not from a
        fixed pixel count.  A fixed count does not scale with the drawing: when
        the fingers were lengthened the constant stayed put, so they came out
        as wire and the hand read as a stick figure.
        """
        if count <= 0:
            return []
        base = (
            MIDDLE_FINGER_LENGTH * FINGER_THICKNESS_RATIO
            * WIDTH_SCALE.get(finger, 1.0)
            * self._scale
        )
        taper = THICKNESS_TAPER
        return [
            max(2, int(round(base * (1.0 - (1.0 - taper) * i / max(1, count - 1)))))
            for i in range(count)
        ]

    def _draw_legend(self, canvas: np.ndarray, wrist_angle: float) -> None:
        y = self.height - 6
        cv2.putText(
            canvas,
            f"wrist_rot {wrist_angle:.0f} deg",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            TEXT_DIM,
            1,
        )


def summarize_pose(angles: Dict[str, float], fingers: Sequence[str] = ()) -> str:
    """One-line text summary of a pose, for logs and tests."""
    names = list(fingers) if fingers else sorted(angles)
    return " ".join(f"{name}={float(angles.get(name, 0.0)):.0f}" for name in names)
