"""Protocol tests for simulated serial communication."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.motor_control.serial_controller import (
    RealSerial,
    SerialController,
    SimulatedSerial,
    is_measured_report,
    is_ping_response,
    parse_ack,
    parse_ack_count,
    parse_measured_angles,
    split_sequence,
)


def test_simulated_ping_matches_firmware_protocol():
    controller = SerialController(SimulatedSerial())
    assert controller.connect()
    assert controller.send_and_receive("PING") == "PONG"
    assert is_ping_response("PONG")


def test_simulated_command_returns_updated_count():
    controller = SerialController(SimulatedSerial())
    controller.connect()
    response = controller.send_and_receive("index_mcp:90,index_pip:100")

    assert response == "OK:2 servos updated"
    assert parse_ack_count(response) == 2
    assert parse_ack_count("OK:1 servos updated") == 1
    assert parse_ack_count("OK:bad") is None


def test_simulated_command_rejects_firmware_invalid_frames():
    invalid_commands = (
        "unknown_joint:90",
        "index_mcp:bad",
        "index_mcp:181",
        "index_mcp:90,index_mcp:100",
        "index_mcp:90:100",
        ",index_mcp:90",
        "index_mcp:90,",
        "index_mcp:90,,index_pip:100",
    )

    for command in invalid_commands:
        controller = SerialController(SimulatedSerial())
        assert controller.connect()
        # send_and_receive fails closed because ERROR is not an ACK.
        assert controller.send_and_receive(command) is None


def test_parse_ack_count_rejects_zero():
    assert parse_ack_count("OK:0 servos updated") is None


def test_parse_ack_exposes_sequence_id():
    assert parse_ack("OK:2 servos updated") == (None, 2)
    assert parse_ack("OK:7:2 servos updated") == (7, 2)
    assert parse_ack_count("OK:7:2 servos updated") == 2
    assert parse_ack("OK:7:0 servos updated") is None
    assert parse_ack("OK::2 servos updated") is None


def test_split_sequence_recognizes_only_the_wire_format():
    assert split_sequence("#12 index_mcp:90") == (12, "index_mcp:90")
    assert split_sequence("index_mcp:90") == (None, "index_mcp:90")
    # A prefix without the separating space is not a sequence id.
    assert split_sequence("#12index_mcp:90") == (None, "#12index_mcp:90")


def test_simulated_ack_echoes_the_sequence_id():
    controller = SerialController(SimulatedSerial())
    assert controller.connect()

    response = controller.send_and_receive("index_mcp:90", seq=12)

    assert response == "OK:12:1 servos updated"
    assert parse_ack(response) == (12, 1)


def test_stale_ack_with_a_different_sequence_is_not_accepted():
    """An ACK that does not echo our id must not be treated as our ACK."""

    class StaleAck(SimulatedSerial):
        """Replays the previous transaction's acknowledgment."""

        def read_line(self):
            return "OK:1:1 servos updated"

    controller = SerialController(StaleAck())
    controller.connect()
    assert controller.send_and_receive("index_mcp:90", timeout=0.05, seq=2) is None
    # The same stale line is accepted when we did not ask for correlation.
    assert controller.send_and_receive("index_mcp:90", timeout=0.05) == "OK:1:1 servos updated"


class _FakePyserialPort:
    """Minimal stand-in for a pyserial port object."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.is_open = True
        self.written = []

    @property
    def in_waiting(self):
        return len(self._lines)

    def readline(self):
        return self._lines.pop(0)

    def write(self, data):
        self.written.append(data)
        return len(data)

    def close(self):
        self.is_open = False


def test_real_serial_strips_only_the_line_terminator():
    """Leading/interior spaces belong to the wire format and must survive."""
    port = RealSerial(port="COM_TEST")
    port._serial = _FakePyserialPort([b"OK:1 servos updated\r\n"])

    assert port.read_line() == "OK:1 servos updated"


def test_real_serial_preserves_a_leading_space():
    port = RealSerial(port="COM_TEST")
    port._serial = _FakePyserialPort([b"  padded \r\n"])

    assert port.read_line() == "  padded "


def test_a_space_padded_ack_is_still_not_a_valid_frame():
    """A frame carrying stray spaces must not be mistaken for a real ACK."""
    port = RealSerial(port="COM_TEST")
    port._serial = _FakePyserialPort([b"OK:1 servos updated \r\n"])

    line = port.read_line()

    assert line == "OK:1 servos updated "
    assert parse_ack(line) is None


# ---------------------------------------------------------------------------
# Measured-angle reports (FOC firmware only)
#
# A hobby servo reports nothing, so on the Arduino build every one of these
# returns None.  They exist for the STM32/SimpleFOC build, where the encoder
# makes the joint angle a MEASURED quantity instead of a commanded one.
# ---------------------------------------------------------------------------

def test_measured_report_parses_a_single_axis():
    assert parse_measured_angles("A:index_mcp:87.3") == {"index_mcp": 87.3}


def test_measured_report_parses_several_axes():
    assert parse_measured_angles("A:index_mcp:87.3,index_pip:91.0") == {
        "index_mcp": 87.3,
        "index_pip": 91.0,
    }


def test_measured_report_keeps_fractional_precision():
    """An encoder reports fractional degrees; rounding loses the point."""
    parsed = parse_measured_angles("A:wrist_rot:-12.5")

    assert parsed == {"wrist_rot": -12.5}
    assert isinstance(parsed["wrist_rot"], float)


def test_measured_report_accepts_an_integer_value():
    assert parse_measured_angles("A:index_mcp:87") == {"index_mcp": 87.0}


def test_measured_report_rejects_a_duplicate_axis():
    assert parse_measured_angles("A:index_mcp:87.3,index_mcp:88.0") is None


def test_measured_report_rejects_an_empty_report():
    """An empty report is not "measured zero axes" -- it is unusable."""
    assert parse_measured_angles("A:") is None


@pytest.mark.parametrize("line", [
    "OK:1 servos updated",
    "PONG",
    "ERROR: bad angle 'x'",
    "A:index_mcp:abc",
    "A:bad name:12",
    "A:index_mcp:",
    "A",
    "",
])
def test_non_reports_are_not_measured_angles(line):
    assert parse_measured_angles(line) is None


def test_measured_parser_rejects_non_strings():
    assert parse_measured_angles(None) is None
    assert parse_measured_angles(87.3) is None


def test_is_measured_report_distinguishes_reports_from_acks():
    assert is_measured_report("A:index_mcp:87.3")
    assert not is_measured_report("OK:1 servos updated")
    assert not is_measured_report("PONG")


def test_arduino_simulator_never_produces_a_measured_report():
    """The servo build has no feedback, so Q must not appear to work.

    This is the whole reason for moving to FOC: if the simulated Arduino
    backend ever answered Q, the Python side would believe it had joint
    feedback it does not have.
    """
    controller = SerialController(SimulatedSerial())
    assert controller.connect()
    response = controller.send_and_receive("Q", timeout=0.2)

    assert parse_measured_angles(response) is None
