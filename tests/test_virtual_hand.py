"""Tests for the virtual hand renderer.

These are geometry tests, not pixel tests.  The bugs this renderer actually had
were all directional -- fingers swinging the wrong way, a thumb that could not
reach the index, a palm that did not rotate with the digits -- and every one of
them was invisible in a still image while being obvious in the numbers.
"""

import math

import numpy as np
import pytest

from src.ui.virtual_hand import VirtualHandRenderer

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
LONG_FINGERS = ("index", "middle", "ring", "pinky")


def extended(value: float = 175.0) -> dict:
    """A relaxed open hand."""
    angles = {f"{f}_{j}": value for f in LONG_FINGERS for j in ("mcp", "pip")}
    angles.update({"thumb_cmc": 155.0, "thumb_mcp": 170.0, "thumb_ip": 170.0})
    return angles


def closed(value: float = 20.0) -> dict:
    """A fist."""
    angles = {f"{f}_{j}": value for f in LONG_FINGERS for j in ("mcp", "pip")}
    angles.update({"thumb_cmc": 45.0, "thumb_mcp": 30.0, "thumb_ip": 25.0})
    return angles


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@pytest.fixture
def renderer() -> VirtualHandRenderer:
    return VirtualHandRenderer(width=420, height=350)


# ---------------------------------------------------------------------------
# Rendering contract
# ---------------------------------------------------------------------------

def test_render_returns_a_canvas_of_the_requested_size(renderer):
    image = renderer.render(extended())

    assert isinstance(image, np.ndarray)
    assert image.shape == (350, 420, 3)
    assert image.dtype == np.uint8


def test_render_actually_draws_something(renderer):
    """A blank canvas has the right shape and would pass a naive test."""
    image = renderer.render(extended())

    assert int(image.std()) > 5, "canvas is a flat colour -- nothing was drawn"


def test_render_accepts_an_empty_mapping(renderer):
    """The dashboard passes {} before the first detection arrives."""
    image = renderer.render({})

    assert image.shape == (350, 420, 3)
    assert int(image.std()) > 5


def test_render_is_deterministic(renderer):
    """The same pose must not jitter between frames."""
    first = renderer.render(extended())
    second = renderer.render(extended())

    assert np.array_equal(first, second)


def test_every_digit_is_solved(renderer):
    pose = renderer._solve(extended())

    assert set(pose) == set(FINGERS)
    for finger, chain in pose.items():
        assert len(chain) >= 4, f"{finger} chain too short: {chain}"
        for point in chain:
            assert all(math.isfinite(v) for v in point), f"{finger} has a non-finite point"


# ---------------------------------------------------------------------------
# Flexion direction -- the bug that made a fist splay sideways
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("finger", FINGERS)
def test_flexion_moves_the_tip_toward_the_palm(renderer, finger):
    palm = renderer._palm_center

    open_tip = renderer._solve(extended())[finger][-1]
    fist_tip = renderer._solve(closed())[finger][-1]

    assert distance(fist_tip, palm) < distance(open_tip, palm), (
        f"{finger} did not fold toward the palm when closed"
    )


def test_a_fist_is_shorter_than_an_open_hand(renderer):
    def span(pose):
        points = [p for chain in pose.values() for p in chain]
        return max(p[1] for p in points) - min(p[1] for p in points)

    assert span(renderer._solve(closed())) < span(renderer._solve(extended()))


def test_long_fingers_do_not_splay_past_the_palm_edges(renderer):
    """A finger must not swing out past the side of the hand.

    An earlier version subtracted the raw flexion angle from the "up" direction,
    which threw every finger bodily sideways -- the index swept from the left of
    the palm to the far right.  The bound is the palm's own half-width plus a
    margin: fingers spread outward when open, so the outermost one legitimately
    reaches the palm edge, but nothing should reach twice that.
    """
    palm_x, _ = renderer._palm_center
    # Use the drawn palm's half-width, not the half-height: the palm is a
    # rounded trapezoid and the fingers root on its top edge.
    half_width = renderer._palm_half_width
    left = palm_x - half_width * 1.8
    right = palm_x + half_width * 1.8

    for pose in (extended(), closed()):
        solved = renderer._solve(pose)
        for finger in LONG_FINGERS:
            for x, _y in solved[finger]:
                assert left <= x <= right, (
                    f"{finger} swung to x={x:.0f}, outside [{left:.0f}, {right:.0f}]"
                )


def test_finger_order_is_preserved(renderer):
    """Fingers converge when a hand closes; they must not cross."""
    for pose in (extended(), closed()):
        solved = renderer._solve(pose)
        tips = [solved[f][-1][0] for f in LONG_FINGERS]
        assert tips == sorted(tips), f"finger order scrambled: {tips}"


# ---------------------------------------------------------------------------
# Thumb -- the axis the user specifically asked to fix
# ---------------------------------------------------------------------------

def test_thumb_tucks_inward_when_the_cmc_axis_flexes(renderer):
    """CMC flexed must bring the thumb toward the index, not away from it."""
    index_tip = renderer._solve(extended())["index"][-1]

    flexed = renderer._solve(dict(extended(), thumb_cmc=0))["thumb"][-1]
    splayed = renderer._solve(dict(extended(), thumb_cmc=180))["thumb"][-1]

    assert distance(flexed, index_tip) < distance(splayed, index_tip), (
        "thumb moves away from the index when CMC flexes -- direction inverted"
    )


def test_thumb_can_reach_the_index_tip(renderer):
    """A pinch has to be drawable, or the operator cannot see one.

    Two earlier versions failed this.  The first could not close at all: the
    thumb started ~93 units from the index tip while the whole thumb was ~66
    long, so no command angle reached.  The second calibrated the flex curve
    against the protocol's 0-180 range, which put a pinch pose outside the
    range the mapper ever emits.

    All three thumb axes are searched, and the index pose is one the mapper
    actually produces (see EXTENDED_ANGLE / FLEXED_ANGLE), because a pinch is a
    coordinated pose.
    """
    index_flexed = {**extended(), "index_mcp": 110.0, "index_pip": 100.0}
    index_tip = renderer._solve(index_flexed)["index"][-1]

    closest = min(
        distance(
            renderer._solve(dict(index_flexed, thumb_cmc=cmc,
                                 thumb_mcp=mcp, thumb_ip=ip))["thumb"][-1],
            index_tip,
        )
        for cmc in range(0, 181, 5)
        for mcp in range(0, 181, 10)
        for ip in range(0, 181, 10)
    )

    # Fingertips are drawn about 5 units across, so under ~10 reads as touching.
    assert closest < 12, f"thumb cannot reach the index tip (closest {closest:.1f})"


def test_thumb_roots_on_the_lateral_side_of_the_palm(renderer):
    """The thumb belongs on the side of the palm, not on the knuckle line."""
    palm_x, _ = renderer._palm_center

    thumb_root = renderer._knuckle("thumb")
    index_root = renderer._knuckle("index")

    assert thumb_root[0] < palm_x, "thumb must root left of the palm centre"
    assert index_root[0] < palm_x + 1, "index must root near the palm centre"


# ---------------------------------------------------------------------------
# Wrist rotation
# ---------------------------------------------------------------------------

def test_wrist_rotation_moves_the_hand(renderer):
    upright = renderer._apply_wrist_rotation(renderer._solve(extended()), 90.0)
    tilted = renderer._apply_wrist_rotation(renderer._solve(extended()), 140.0)

    assert distance(upright["middle"][-1], tilted["middle"][-1]) > 10


def test_wrist_rotation_is_rigid(renderer):
    """Rotation must not stretch the hand: bone lengths are preserved."""
    upright = renderer._apply_wrist_rotation(renderer._solve(extended()), 90.0)
    tilted = renderer._apply_wrist_rotation(renderer._solve(extended()), 140.0)

    for finger in FINGERS:
        for i in range(len(upright[finger]) - 1):
            before = distance(upright[finger][i], upright[finger][i + 1])
            after = distance(tilted[finger][i], tilted[finger][i + 1])
            assert before == pytest.approx(after, abs=1e-6), (
                f"{finger} segment {i} changed length under rotation"
            )


def test_wrist_rotation_leaves_the_pivot_fixed(renderer):
    rotated = renderer._rotate_point(renderer._wrist, 137.0)

    assert rotated[0] == pytest.approx(renderer._wrist[0], abs=1e-6)
    assert rotated[1] == pytest.approx(renderer._wrist[1], abs=1e-6)


def test_palm_rotates_with_the_digits(renderer):
    """The palm is drawn through the same transform as the fingers.

    Drawing the palm unrotated while the digits rotated left every finger
    rooted in mid-air beside a stationary palm plate.
    """
    wrist_angle = 140.0
    palm_centre = renderer._palm_center
    rotated_centre = renderer._rotate_point(palm_centre, wrist_angle)

    pose = renderer._apply_wrist_rotation(renderer._solve(extended()), wrist_angle)

    # Every digit root must still sit on the rotated palm's rim.
    for finger, chain in pose.items():
        gap = distance(chain[0], rotated_centre)
        assert gap <= renderer._palm_radius * 1.15, (
            f"{finger} root is {gap:.0f} from the rotated palm centre"
        )


@pytest.mark.parametrize("wrist_angle", [0, 30, 90, 150, 180])
def test_nothing_is_drawn_off_canvas(renderer, wrist_angle):
    """A clipped fingertip is worse than a smaller hand: the operator cannot
    tell 'at the limit' from 'the drawing broke'."""
    angles = {**extended(), "wrist_rot": float(wrist_angle)}
    pose = renderer._apply_wrist_rotation(renderer._solve(angles), float(wrist_angle))

    for finger, chain in pose.items():
        for x, y in chain:
            cx, cy = renderer._to_canvas((x, y))
            assert 0 <= cx < renderer.width, f"{finger} x={cx} off canvas"
            assert 0 <= cy < renderer.height, f"{finger} y={cy} off canvas"


# ---------------------------------------------------------------------------
# Flex-curve calibration
#
# The bug these guard against: the drawing was calibrated against the protocol's
# 0-180 range, but the mapper only ever emits roughly 68-158.  A real hand
# closing therefore used just the middle of the curve and moved the fingertip
# ~12 units out of ~54 -- which looked exactly like the sketch ignoring the
# hand.  These tests fail if the calibration drifts back off the real range.
# ---------------------------------------------------------------------------

def test_flex_curve_matches_the_mappers_real_range():
    """The drawing must be calibrated to the angles the mapper emits."""
    from src.ui.virtual_hand import EXTENDED_ANGLE, FLEXED_ANGLE

    assert EXTENDED_ANGLE == 158.0
    assert FLEXED_ANGLE == 68.0
    # Calibrating to 0-180 is the specific mistake this guards.
    assert EXTENDED_ANGLE < 180.0 and FLEXED_ANGLE > 0.0


def test_fold_fraction_spans_the_working_range(renderer):
    """A straight hand reads 0, a closed hand reads 1 -- within real angles."""
    from src.ui.virtual_hand import EXTENDED_ANGLE, FLEXED_ANGLE

    assert renderer._fold_fraction(EXTENDED_ANGLE) == pytest.approx(0.0)
    assert renderer._fold_fraction(FLEXED_ANGLE) == pytest.approx(1.0)
    # Anything beyond the working range saturates rather than overshooting.
    assert renderer._fold_fraction(180.0) == pytest.approx(0.0)
    assert renderer._fold_fraction(0.0) == pytest.approx(1.0)


def test_a_real_hand_closing_moves_the_fingertip_substantially(renderer):
    """The whole point of the view: the sketch must visibly follow the hand.

    Uses the mapper's real output range, not the protocol's.  An earlier
    version moved the tip only ~12 units here, which is why the renderer looked
    like it was ignoring the hand.
    """
    palm = renderer._palm_center

    def tip_at(mcp, pip):
        pose = {**extended(), "index_mcp": mcp, "index_pip": pip}
        return renderer._solve(pose)["index"][-1]

    open_tip = tip_at(158.0, 180.0)
    closed_tip = tip_at(68.0, 90.0)

    travel = distance(open_tip, closed_tip)
    assert travel > 25.0, (
        f"fingertip only travelled {travel:.1f} units across the mapper's real "
        "range -- the sketch will look like it is ignoring the hand"
    )


def test_fingertip_approaches_the_palm_monotonically(renderer):
    """Every step of a closing hand must move the tip closer, never away."""
    palm = renderer._palm_center
    previous = None

    for mcp in (158.0, 138.0, 118.0, 98.0, 78.0, 68.0):
        pip = 90.0 + (mcp - 68.0)
        pose = {**extended(), "index_mcp": mcp, "index_pip": pip}
        tip = renderer._solve(pose)["index"][-1]
        current = distance(tip, palm)

        if previous is not None:
            assert current < previous + 0.5, (
                f"tip moved away from the palm at mcp={mcp}: "
                f"{previous:.1f} -> {current:.1f}"
            )
        previous = current


# ---------------------------------------------------------------------------
# Finger curl direction
#
# The bug these guard against: every finger rotated toward a shared target
# direction, so the outer fingers overshot and crossed over their neighbours --
# a fist drew with the index right of the ring finger.  A real hand curls all
# four the same rotational sense; because they start spread they stay spread.
# ---------------------------------------------------------------------------

def test_all_fingers_curl_the_same_rotational_sense(renderer):
    """Opposite signs make fingers converge and cross.  They must not."""
    from src.ui.virtual_hand import _flex_sign

    signs = {_flex_sign(f) for f in LONG_FINGERS}
    assert signs == {1.0}, (
        f"fingers curl in different senses {signs}; they will cross over"
    )


@pytest.mark.parametrize("pose_name", ["extended", "closed"])
def test_fingers_never_cross(renderer, pose_name):
    """Fingertips keep their left-to-right order through the whole curl."""
    pose = extended() if pose_name == "extended" else closed()
    solved = renderer._solve(pose)

    tips = [solved[f][-1][0] for f in LONG_FINGERS]
    assert tips == sorted(tips), (
        f"{pose_name}: finger order scrambled {tips} -- fingers crossed"
    )


def test_fingers_stay_ordered_partway_through_the_curl(renderer):
    """Crossing can appear mid-curl, not only at the extremes."""
    for mcp in range(68, 159, 10):
        pip = min(180.0, mcp + 12.0)
        pose = {**extended(), "index_mcp": float(mcp), "index_pip": float(pip),
                "middle_mcp": float(mcp), "middle_pip": float(pip),
                "ring_mcp": float(mcp), "ring_pip": float(pip),
                "pinky_mcp": float(mcp), "pinky_pip": float(pip)}
        solved = renderer._solve(pose)
        tips = [solved[f][-1][0] for f in LONG_FINGERS]
        assert tips == sorted(tips), f"crossed at mcp={mcp}: {tips}"


def test_a_finger_does_not_swing_across_to_the_far_side(renderer):
    """A closing index must stay on its own side of the hand.

    An earlier version swept it from x=134 out to x=186 -- past the middle and
    ring fingers and over to the far edge of the hand.

    The bound is the palm's half-width, not its centreline: a finger curling
    toward the palm legitimately sweeps a few units past the centre, and
    demanding it stay strictly left was too strict.  What must never happen is
    reaching the far side of the hand.
    """
    palm_x, _ = renderer._palm_center
    half_width = renderer._palm_half_width

    for mcp in range(68, 159, 10):
        pose = {**extended(), "index_mcp": float(mcp), "index_pip": float(mcp)}
        tip = renderer._solve(pose)["index"][-1]
        assert tip[0] < palm_x + half_width, (
            f"index swung to x={tip[0]:.0f}, past the far edge of the palm "
            f"({palm_x + half_width:.0f})"
        )
