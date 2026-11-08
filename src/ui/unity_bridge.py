"""
Unity pose bridge.

Streams the solved joint angles to a Unity client over UDP so that a 3D hand
model can mirror the tracked hand.  Unity is only ever a *consumer* of the pose:
the vision, the mapping and the safety logic all stay on this side, so a Unity
client that is closed, crashed or not yet written cannot affect the control
loop.

Why UDP rather than TCP
-----------------------
This is a live pose stream, not a reliable message channel.  A dropped frame
costs the model one frame time and the next packet supersedes it.  TCP would
instead buy retransmission and head-of-line blocking, which let latency
accumulate on a stream where old data is worthless.  On loopback the loss rate
is effectively zero anyway.

Why the array order is part of the contract
-------------------------------------------
Unity's built-in JSON reader (``JsonUtility``) cannot deserialise a dictionary,
so the angles travel as a fixed-order array rather than as a name->value map.
That makes the order part of the wire format: if the two sides ever disagreed,
the model would drive the wrong fingers and *nothing would report an error*.
``AXIS_ORDER`` is therefore asserted against the project's canonical joint order
in the tests, and a "hello" packet carries the names so the Unity side can log
exactly what it is about to receive.

Wire format
-----------
Sent once on start::

    {"hello": true, "n": ["thumb_cmc", "thumb_mcp", ...]}

Sent per frame::

    {"t": 12.345, "a": [90.0, 90.0, ...], "g": "ok", "o": "back", "oc": 0.92}

``a``    joint angles in ``AXIS_ORDER``, degrees
``t``    seconds since the bridge started
``g``    recognised gesture name, or absent when none is recognised
``o``    hand orientation: "front" | "back" | "side" | "unknown"
``oc``   orientation confidence, 0..1
"""

import json
import logging
import socket
import time
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Canonical joint order.  MUST match JointAngles field order and the firmware's
# SERVO_NAMES; tests/test_unity_bridge.py enforces this.  Do not reorder.
AXIS_ORDER: tuple = (
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp",
    "index_pip",
    "middle_mcp",
    "middle_pip",
    "ring_mcp",
    "ring_pip",
    "pinky_mcp",
    "pinky_pip",
    "wrist_rot",
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888

# How often to repeat the hello packet.  It carries the axis order, and the
# receiver needs it to map angles to joints.  Sending it only once means a
# Unity session that starts playing afterwards never learns the order: the
# poses arrive, the status HUD says "live", and nothing moves -- a silent
# failure.  Repeating it costs ~60 bytes per second and removes that race.
HELLO_RESEND_SECONDS = 1.0

# A single UDP datagram carrying one pose.  Well under the 1500-byte path MTU,
# so it will not be fragmented on a typical network.
_MAX_PACKET_BYTES = 1024


def pose_payload(
    angles: Dict[str, float],
    elapsed: float,
    gesture: Optional[str] = None,
    orientation: Optional[str] = None,
    orientation_confidence: Optional[float] = None,
    motion: Optional[str] = None,
) -> str:
    """Build one pose packet as JSON text.

    Missing axis names fall back to the neutral 90 degrees rather than raising:
    a partially-solved frame should still move the model, and refusing to send
    anything would freeze the display on stale data.
    """
    ordered: List[float] = []
    for name in AXIS_ORDER:
        value = angles.get(name, 90.0)
        try:
            ordered.append(round(float(value), 2))
        except (TypeError, ValueError):
            ordered.append(90.0)

    payload: Dict[str, object] = {"t": round(float(elapsed), 3), "a": ordered}
    if gesture:
        payload["g"] = str(gesture)
    if orientation:
        payload["o"] = str(orientation)
    if orientation_confidence is not None:
        payload["oc"] = round(float(orientation_confidence), 3)
    if motion:
        payload["m"] = str(motion)
    return json.dumps(payload, separators=(",", ":"))


def hello_payload() -> str:
    """Build the handshake packet that tells Unity the axis order."""
    return json.dumps({"hello": True, "n": list(AXIS_ORDER)}, separators=(",", ":"))


class UnityPoseStream:
    """Send joint angles to a Unity client over UDP.

    Usage::

        stream = UnityPoseStream()
        stream.start()
        ...
        stream.send(angles_dict, gesture="ok", orientation="back")
        stream.close()

    Every send is best-effort.  A socket error is logged once and then the
    bridge disables itself, so a missing Unity client cannot fill the log or
    stall the control loop.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        enabled: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.enabled = bool(enabled)
        self._socket: Optional[socket.socket] = None
        self._start_time = 0.0
        self._last_hello = 0.0
        self._sent = 0
        self._failed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Create the socket and announce the axis order.  Returns success."""
        if not self.enabled:
            return False
        if self._socket is not None:
            return True
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Never block waiting to flush: the control loop must not stall on
            # the network even if a host is unreachable.
            self._socket.setblocking(False)
            self._start_time = time.monotonic()
            self._raw_send(hello_payload())
            logger.info("Unity pose stream started -> %s:%d", self.host, self.port)
            return True
        except OSError as exc:
            logger.warning("Unity pose stream unavailable (%s); continuing without it", exc)
            self._disable()
            return False

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    @property
    def is_running(self) -> bool:
        return self._socket is not None and not self._failed

    @property
    def packets_sent(self) -> int:
        return self._sent

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send(
        self,
        angles: Dict[str, float],
        gesture: Optional[str] = None,
        orientation: Optional[str] = None,
        orientation_confidence: Optional[float] = None,
        motion: Optional[str] = None,
    ) -> bool:
        """Send one pose.  Returns whether it went out."""
        if not self.is_running:
            return False
        elapsed = time.monotonic() - self._start_time

        # Re-announce the axis order periodically; see HELLO_RESEND_SECONDS.
        if elapsed - self._last_hello >= HELLO_RESEND_SECONDS:
            self._raw_send(hello_payload())
            self._last_hello = elapsed

        return self._raw_send(
            pose_payload(angles, elapsed, gesture, orientation,
                         orientation_confidence, motion)
        )

    def _raw_send(self, text: str) -> bool:
        """Encode and transmit one packet, failing safe."""
        if self._socket is None:
            return False
        data = text.encode("utf-8")
        if len(data) > _MAX_PACKET_BYTES:  # pragma: no cover - guarded by design
            logger.error("Unity packet too large (%d bytes); dropped", len(data))
            return False
        try:
            self._socket.sendto(data, (self.host, self.port))
            self._sent += 1
            return True
        except OSError as exc:
            # A closed client port on loopback surfaces as ECONNREFUSED on some
            # platforms.  That is not a reason to keep trying forever.
            logger.warning("Unity pose send failed (%s); disabling the stream", exc)
            self._disable()
            return False

    def _disable(self) -> None:
        self._failed = True
        self.close()
