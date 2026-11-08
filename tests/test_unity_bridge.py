"""Tests for the Unity pose bridge.

The load-bearing test here is the axis-order check.  Everything else on the wire
fails loudly if it is wrong -- malformed JSON is rejected, a missing packet is a
timeout -- but a silently reordered angle array just drives the wrong fingers on
the model and reports nothing at all.  That is the failure this file exists to
prevent.
"""

import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_mapper.joint_mapper import JointAngles
from src.ui.unity_bridge import (
    AXIS_ORDER,
    UnityPoseStream,
    hello_payload,
    pose_payload,
)


# ---------------------------------------------------------------------------
# The wire contract
# ---------------------------------------------------------------------------

def test_axis_order_matches_the_project_canonical_order():
    """The array order must equal the joint order used everywhere else.

    If this ever fails, do NOT reorder AXIS_ORDER to match -- work out which
    side changed and why.  Reordering silently re-routes every angle on the
    wire.
    """
    assert AXIS_ORDER == tuple(JointAngles().to_dict().keys())


def test_axis_order_has_no_duplicates():
    assert len(AXIS_ORDER) == len(set(AXIS_ORDER))


def test_axis_order_covers_all_twelve_axes():
    assert len(AXIS_ORDER) == 12


def test_pose_payload_emits_angles_in_axis_order():
    angles = {name: float(i) for i, name in enumerate(AXIS_ORDER)}

    payload = json.loads(pose_payload(angles, elapsed=1.0))

    assert payload["a"] == [float(i) for i in range(12)]


def test_pose_payload_reorders_a_shuffled_mapping():
    """A dict built in a different order must still send in AXIS_ORDER."""
    shuffled = {name: float(i) for i, name in reversed(list(enumerate(AXIS_ORDER)))}

    payload = json.loads(pose_payload(shuffled, elapsed=1.0))

    assert payload["a"] == [float(i) for i in range(12)]


def test_pose_payload_fills_missing_axes_with_neutral():
    """A partially solved frame should still move the model, not freeze it."""
    payload = json.loads(pose_payload({"index_mcp": 30.0}, elapsed=1.0))

    values = payload["a"]
    assert len(values) == 12
    assert values[AXIS_ORDER.index("index_mcp")] == 30.0
    assert values[AXIS_ORDER.index("thumb_cmc")] == 90.0


def test_pose_payload_survives_a_non_finite_angle():
    payload = json.loads(pose_payload({"index_mcp": float("nan")}, elapsed=1.0))

    # NaN is not valid JSON; it must be replaced rather than emitted.
    assert all(isinstance(v, float) for v in payload["a"])
    assert json.loads(pose_payload({"index_mcp": float("inf")}, 1.0))


def test_hello_payload_carries_the_names():
    payload = json.loads(hello_payload())

    assert payload["hello"] is True
    assert payload["n"] == list(AXIS_ORDER)


def test_pose_payload_omits_absent_metadata():
    """No gesture recognised is different from a gesture named empty string."""
    payload = json.loads(pose_payload({}, elapsed=0.0))

    assert "g" not in payload
    assert "o" not in payload


def test_pose_payload_includes_metadata_when_present():
    payload = json.loads(
        pose_payload({}, elapsed=0.0, gesture="ok", orientation="back",
                     orientation_confidence=0.9)
    )

    assert payload["g"] == "ok"
    assert payload["o"] == "back"
    assert payload["oc"] == 0.9


# ---------------------------------------------------------------------------
# Transport behaviour
# ---------------------------------------------------------------------------

class _RecordingSocket:
    """Stand-in for a UDP socket that records what was sent."""

    def __init__(self):
        self.sent = []

    def setblocking(self, flag):
        pass

    def sendto(self, data, address):
        self.sent.append((data, address))
        return len(data)

    def close(self):
        pass


@pytest.fixture
def loopback():
    """A real UDP socket bound to an ephemeral loopback port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.5)
    yield sock
    sock.close()


def test_stream_sends_a_hello_then_poses(loopback):
    port = loopback.getsockname()[1]
    stream = UnityPoseStream(port=port)
    assert stream.start()

    stream.send({name: 45.0 for name in AXIS_ORDER}, gesture="ok")

    hello, _ = loopback.recvfrom(2048)
    assert json.loads(hello)["hello"] is True
    pose, _ = loopback.recvfrom(2048)
    assert json.loads(pose)["g"] == "ok"

    stream.close()


def test_disabled_stream_sends_nothing(loopback):
    port = loopback.getsockname()[1]
    stream = UnityPoseStream(port=port, enabled=False)

    assert stream.start() is False
    assert stream.send({}) is False
    assert stream.packets_sent == 0


def test_send_before_start_is_a_no_op():
    stream = UnityPoseStream()

    assert stream.send({}) is False
    assert stream.is_running is False


def test_a_socket_error_disables_the_stream_instead_of_raising():
    """A missing Unity client must never disturb the control loop."""

    class _FailingSocket(_RecordingSocket):
        def sendto(self, data, address):
            raise OSError("connection refused")

    stream = UnityPoseStream()
    stream._socket = _FailingSocket()
    stream._start_time = 0.0

    assert stream.send({}) is False          # fails, does not raise
    assert stream.is_running is False
    assert stream.send({}) is False          # and stays disabled


def test_close_is_idempotent():
    stream = UnityPoseStream(port=9)
    stream.start()
    stream.close()
    stream.close()

    assert stream.is_running is False


def test_packets_stay_within_the_datagram_budget(loopback):
    """A pose packet must fit one datagram without fragmenting."""
    port = loopback.getsockname()[1]
    stream = UnityPoseStream(port=port)
    stream.start()

    stream.send({name: 179.99 for name in AXIS_ORDER},
                gesture="pinch", orientation="unknown", orientation_confidence=1.0)

    loopback.recvfrom(2048)                  # hello
    pose, _ = loopback.recvfrom(2048)
    assert len(pose) < 512

    stream.close()


def test_hello_is_resent_so_a_late_client_can_join(loopback):
    """The axis order must reach a Unity session that started after us.

    The hello packet carries the joint order, and the receiver needs it to map
    angles to joints.  Sending it only at startup means a client that begins
    playing later never learns the order: poses arrive, the status HUD reports
    "live", and nothing moves.  This is the exact failure that was observed, so
    the resend is covered rather than assumed.
    """
    port = loopback.getsockname()[1]
    stream = UnityPoseStream(port=port)
    assert stream.start()

    stream.send({})                          # startup hello, then a pose
    loopback.recvfrom(2048)                  # hello
    loopback.recvfrom(2048)                  # pose

    stream._last_hello = -1.0                # pretend the interval elapsed
    stream.send({})

    first, _ = loopback.recvfrom(2048)
    assert json.loads(first)["hello"] is True
    assert json.loads(first)["n"] == list(AXIS_ORDER)

    stream.close()
