"""
Serial Communication Controller.

Handles communication with the Arduino microcontroller via serial (UART).
Provides both real and simulated serial backends for development/testing.

Protocol:
  - Commands sent as ASCII text: "joint:angle,joint:angle,...\\n"
  - Arduino echoes back received angles for verification
  - Heartbeat monitoring for connection health
"""

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


_ACK_PATTERN = re.compile(r"^OK:(?:(\d{1,3}):)?(\d{1,3})\s+servos updated$")
_SEQ_PREFIX_PATTERN = re.compile(r"^#(\d{1,3}) (.*)$")
_ANGLE_PATTERN = re.compile(r"^[+-]?[0-9]+$")
# Measured-angle report, sent by FOC firmware in reply to "Q":
#   A:index_mcp:87.3,index_pip:91.0
# Hobby servos cannot produce this line at all -- they have no feedback.  The
# firmware only emits it when asked, so its absence is not an error.
_MEASURED_PATTERN = re.compile(
    r"^A:((?:[A-Za-z_][A-Za-z0-9_]*:-?[0-9]+(?:\.[0-9]+)?)(?:,[A-Za-z_][A-Za-z0-9_]*:-?[0-9]+(?:\.[0-9]+)?)*)$"
)
_MEASURED_FIELD_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(-?[0-9]+(?:\.[0-9]+)?)$")
_MAX_COMMAND_PAYLOAD = 255
_MAX_ACK_COUNT = 12
_DEFAULT_SERVO_NAMES = frozenset(
    {
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
    }
)


def _prepare_command(command: str) -> Optional[str]:
    """Validate and normalize one newline-terminated ASCII command."""
    if not isinstance(command, str):
        logger.error("Serial command must be a string")
        return None

    payload = command[:-1] if command.endswith("\n") else command
    if not payload:
        logger.error("Serial command cannot be empty")
        return None
    if any(char in payload for char in ("\r", "\n", "\x00")):
        logger.error("Serial command contains an embedded frame delimiter or NUL")
        return None

    try:
        encoded = payload.encode("ascii")
    except UnicodeEncodeError:
        logger.error("Serial command must contain ASCII characters only")
        return None
    if len(encoded) > _MAX_COMMAND_PAYLOAD:
        logger.error(
            "Serial command is too long (%d bytes; maximum %d)",
            len(encoded),
            _MAX_COMMAND_PAYLOAD,
        )
        return None
    return payload + "\n"


def is_ping_response(response: Optional[str]) -> bool:
    """Return whether a response is the firmware's canonical PONG."""
    return response is not None and response.strip() == "PONG"


def split_sequence(command: str) -> tuple[Optional[int], str]:
    """Split an optional ``#<seq> `` prefix from a command frame.

    The firmware echoes the id on its acknowledgment, which is what makes an
    ACK correlatable: two frames that update the same number of axes are
    otherwise indistinguishable.
    """
    if not isinstance(command, str):
        return None, command
    match = _SEQ_PREFIX_PATTERN.match(command)
    if not match:
        return None, command
    return int(match.group(1)), match.group(2)


def parse_ack(response: Optional[str]) -> Optional[tuple[Optional[int], int]]:
    """Parse an acknowledgment into ``(sequence_id, updated_count)``.

    ``sequence_id`` is ``None`` for a frame sent without one.  A malformed or
    unrelated line returns ``None`` so callers can fail safe instead of
    treating an arbitrary serial line as an acknowledgment.

    Matching is strict: the line terminator must already have been removed by
    the transport.  Accepting padded text would let a corrupted frame through
    as a valid acknowledgment.
    """
    if response is None:
        return None
    match = _ACK_PATTERN.fullmatch(response)
    if not match:
        return None
    try:
        count = int(match.group(2))
    except ValueError:  # pragma: no cover - the regex limits the value size
        return None
    if not 1 <= count <= _MAX_ACK_COUNT:
        return None
    seq = int(match.group(1)) if match.group(1) is not None else None
    return seq, count


def parse_ack_count(response: Optional[str]) -> Optional[int]:
    """Parse ``OK:N servos updated`` and return ``N``.

    The Arduino firmware uses this response for every non-PING command.  A
    malformed or unrelated line returns ``None`` so callers can fail safe
    instead of treating an arbitrary serial line as an acknowledgment.
    """
    ack = parse_ack(response)
    return None if ack is None else ack[1]


def parse_measured_angles(response: Optional[str]) -> Optional[dict]:
    """Parse a measured-angle report into ``{joint_name: degrees}``.

    Only the FOC firmware can answer this, and only when asked with ``Q``: a
    hobby servo reports nothing, so on the Arduino build this returns ``None``
    for every frame.  Returning ``None`` rather than ``{}`` keeps "the firmware
    cannot measure" distinct from "it measured nothing".

    Values are floats, not ints -- an encoder reports fractional degrees, and
    rounding here would throw away the precision that motivated the encoder.
    """
    if not isinstance(response, str):
        return None
    match = _MEASURED_PATTERN.fullmatch(response)
    if not match:
        return None

    angles: dict = {}
    for field in match.group(1).split(","):
        field_match = _MEASURED_FIELD_PATTERN.fullmatch(field)
        if field_match is None:  # pragma: no cover - the outer regex guards this
            return None
        name = field_match.group(1)
        if name in angles:
            return None          # duplicate axis: refuse rather than guess
        try:
            angles[name] = float(field_match.group(2))
        except ValueError:  # pragma: no cover - regex restricts the shape
            return None
    return angles or None


def is_measured_report(response: Optional[str]) -> bool:
    """Whether a line is a measured-angle report rather than an ACK."""
    return parse_measured_angles(response) is not None


def _parse_simulated_command(command: str) -> Optional[int]:
    """Validate a joint frame using the same rules as the Arduino parser."""
    if not command or command in {"PING"}:
        return None

    fields = command.split(",")
    if not fields or any(not field for field in fields):
        return None

    seen = set()
    for field in fields:
        if field.count(":") != 1:
            return None
        name, angle_text = field.split(":", 1)
        if name not in _DEFAULT_SERVO_NAMES or name in seen:
            return None
        if not _ANGLE_PATTERN.fullmatch(angle_text):
            return None
        try:
            angle = int(angle_text, 10)
        except ValueError:  # pragma: no cover - regex guards this path
            return None
        if not 0 <= angle <= 180:
            return None
        seen.add(name)

    return len(fields)


# ---------------------------------------------------------------------------
# Abstract Serial Interface
# ---------------------------------------------------------------------------

class SerialInterface(ABC):
    """Abstract interface for serial communication."""

    @abstractmethod
    def open(self) -> bool:
        """Open the serial connection. Returns True on success."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the serial connection."""
        ...

    @abstractmethod
    def write(self, data: str) -> int:
        """Write a string to the serial port. Returns bytes written."""
        ...

    @abstractmethod
    def read_line(self) -> Optional[str]:
        """Read one line from the serial port. Non-blocking."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the connection is alive."""
        ...


# ---------------------------------------------------------------------------
# Real Serial Implementation (PySerial)
# ---------------------------------------------------------------------------

class RealSerial(SerialInterface):
    """
    Serial communication using physical UART/COM port.

    Usage:
        serial = RealSerial(port="COM3", baudrate=115200)
        serial.open()
        serial.write("index_mcp:90\\n")
        serial.close()
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None

    def open(self) -> bool:
        try:
            import serial
        except ImportError:
            logger.error("PySerial not installed. Run: pip install pyserial")
            return False

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            # Allow Arduino to reset after connection, then discard startup
            # banners so the first PING reads the response to its own command.
            time.sleep(2.0)
            self._serial.reset_input_buffer()
            logger.info("Serial connection opened on %s @ %d baud", self.port, self.baudrate)
            return True
        except serial.SerialException as e:
            logger.error("Failed to open serial port %s: %s", self.port, e)
            return False

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("Serial connection closed")

    def write(self, data: str) -> int:
        if not self.is_connected():
            raise ConnectionError("Serial port not connected")
        encoded = data.encode("ascii")
        return self._serial.write(encoded)

    def read_line(self) -> Optional[str]:
        if not self.is_connected():
            return None
        try:
            if self._serial.in_waiting > 0:
                line = self._serial.readline()
                # Strip only the terminator.  A bare strip() would also eat
                # leading/trailing spaces, which are part of the wire format
                # the parser validates.
                return line.decode("ascii", errors="replace").rstrip("\r\n")
        except Exception as e:
            logger.debug("Serial read error: %s", e)
        return None

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open


# ---------------------------------------------------------------------------
# Simulated Serial (for development without hardware)
# ---------------------------------------------------------------------------

class SimulatedSerial(SerialInterface):
    """
    Simulated serial interface for development and testing.

    Prints commands to console and simulates Arduino echo responses.
    """

    def __init__(self, echo_enabled: bool = True, verbose: bool = False):
        self.echo_enabled = echo_enabled
        self.verbose = verbose
        self._connected = False
        self._last_command = ""
        self._command_count = 0

    def open(self) -> bool:
        self._connected = True
        logger.info("Simulated Serial connected (no hardware)")
        return True

    def close(self) -> None:
        self._connected = False
        logger.info("Simulated Serial disconnected")

    def write(self, data: str) -> int:
        if not self._connected:
            raise ConnectionError("Simulated serial not connected")
        self._last_command = data[:-1] if data.endswith("\n") else data
        self._command_count += 1

        if self.verbose:
            logger.info("[SIM] -> %s", self._last_command)

        return len(data)

    def read_line(self) -> Optional[str]:
        if not self._connected:
            return None

        if self.echo_enabled and self._last_command:
            cmd = self._last_command
            self._last_command = ""
            seq, frame = split_sequence(cmd)
            if frame == "PING":
                return "PONG"
            if frame:
                updated = _parse_simulated_command(frame)
                error_tag = "" if seq is None else f"[{seq}]"
                if updated is None:
                    return f"ERROR{error_tag}: invalid command"
                seq_prefix = "" if seq is None else f"{seq}:"
                return f"OK:{seq_prefix}{updated} servos updated"

        return None

    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# Serial Controller
# ---------------------------------------------------------------------------

class SerialController:
    """
    High-level controller for serial communication with the bionic hand hardware.

    Handles:
      - Connection lifecycle (open/close/reconnect)
      - Command sending with optional acknowledgments
      - Heartbeat monitoring
      - Auto-reconnection on disconnect

    Usage:
        ctrl = SerialController(interface=my_serial)
        ctrl.connect()
        ctrl.send_command("index_mcp:90,index_pip:170")
        ctrl.disconnect()
    """

    RECONNECT_DELAY = 3.0  # Seconds between reconnection attempts

    def __init__(self, interface: SerialInterface):
        """
        Args:
            interface: SerialInterface implementation (real or simulated).
        """
        self._interface = interface
        self._lock = threading.Lock()
        self._last_heartbeat = 0.0
        self._heartbeat_interval = 1.0
        self._reconnect_enabled = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish the serial connection."""
        with self._lock:
            if self._interface.is_connected():
                logger.warning("Already connected")
                return True

            success = self._interface.open()
            if success:
                self._last_heartbeat = time.time()
            return success

    def disconnect(self) -> None:
        """Close the serial connection."""
        self._reconnect_enabled = False
        with self._lock:
            self._interface.close()

    @property
    def is_connected(self) -> bool:
        return self._interface.is_connected()

    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------

    def _write_command_unlocked(self, command: str) -> bool:
        """Validate and write one frame; caller must hold ``_lock``."""
        if not self._interface.is_connected():
            logger.warning("Cannot send command: not connected")
            return False

        try:
            framed = _prepare_command(command)
            if framed is None:
                return False

            expected_bytes = len(framed.encode("ascii"))
            bytes_written = self._interface.write(framed)
            logger.debug("Sent %d bytes: %s", bytes_written, framed.strip())
            if bytes_written != expected_bytes:
                logger.error(
                    "Short serial write: %d/%d bytes", bytes_written, expected_bytes
                )
                return False
            return True

        except (ConnectionError, OSError, UnicodeError) as e:
            logger.error("Serial write failed: %s", e)
            return False

    def _read_response_unlocked(self) -> Optional[str]:
        """Read one response line; caller must hold ``_lock``."""
        try:
            return self._interface.read_line()
        except (ConnectionError, OSError) as exc:
            logger.warning("Serial read failed: %s", exc)
            return None

    def _drain_unlocked(self, limit: int = 64) -> int:
        """Discard already queued asynchronous lines before a transaction."""
        drained = 0
        for _ in range(max(0, limit)):
            response = self._read_response_unlocked()
            if response is None:
                break
            drained += 1
            logger.debug("Discarding stale serial response: %r", response)
        return drained

    def send_command(self, command: str) -> bool:
        """Send one validated command frame without waiting for a response."""
        with self._lock:
            return self._write_command_unlocked(command)

    def read_response(self) -> Optional[str]:
        """Read a response line from the hardware (non-blocking)."""
        with self._lock:
            return self._read_response_unlocked()

    def drain_responses(self, limit: int = 32) -> list[str]:
        """Read up to ``limit`` pending response lines without blocking."""
        responses: list[str] = []
        with self._lock:
            for _ in range(max(0, limit)):
                response = self._read_response_unlocked()
                if response is None:
                    break
                responses.append(response)
        return responses

    def send_and_receive(
        self,
        command: str,
        timeout: float = 0.5,
        seq: Optional[int] = None,
    ) -> Optional[str]:
        """Send a command and wait for its canonical response.

        The write and response wait share one lock so another thread cannot
        interleave a transaction.  Queued asynchronous lines are discarded
        before the write, and unrelated lines (for example a timeout warning)
        are ignored while waiting.

        Pass ``seq`` to prepend a ``#<seq> `` prefix; the response is then only
        accepted when it echoes that same id.  Without a sequence id the text
        protocol cannot distinguish a stale ACK from the expected one, and the
        pre-send drain is the only mitigation.
        """
        if isinstance(command, str):
            payload = command[:-1] if command.endswith("\n") else command
        else:
            payload = command
        expected_ping = payload == "PING"
        framed = f"#{seq} {payload}\n" if seq is not None and isinstance(payload, str) else command
        deadline_timeout = max(0.0, timeout)

        with self._lock:
            if not self._interface.is_connected():
                logger.warning("Cannot send command: not connected")
                return None

            self._drain_unlocked()
            if not self._write_command_unlocked(framed):
                return None

            deadline = time.monotonic() + deadline_timeout
            while time.monotonic() < deadline:
                response = self._read_response_unlocked()
                if response is not None:
                    if expected_ping:
                        matches = is_ping_response(response)
                    else:
                        ack = parse_ack(response)
                        matches = ack is not None and (seq is None or ack[0] == seq)
                    if matches:
                        return response
                    logger.warning("Ignoring unrelated serial response: %r", response)
                time.sleep(0.01)

        logger.debug("Response timeout for command: %s", str(command).strip())
        return None

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(self) -> bool:
        """
        Perform a heartbeat check. Call periodically (e.g., every 1s).

        Returns:
            True if the connection is healthy.
        """
        now = time.time()
        if (now - self._last_heartbeat) < self._heartbeat_interval:
            return self.is_connected

        self._last_heartbeat = now
        if not self._interface.is_connected():
            if self._reconnect_enabled:
                logger.info("Attempting reconnection...")
                if not self._interface.open():
                    return False
            else:
                return False

        response = self.send_and_receive("PING", timeout=0.5)
        return is_ping_response(response)

    def enable_auto_reconnect(self) -> None:
        """Enable automatic reconnection on disconnect."""
        self._reconnect_enabled = True
        logger.info("Auto-reconnect enabled")

    def disable_auto_reconnect(self) -> None:
        """Disable automatic reconnection."""
        self._reconnect_enabled = False
