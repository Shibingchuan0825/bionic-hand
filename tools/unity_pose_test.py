#!/usr/bin/env python3
"""Test tool for the Unity pose link, usable before Unity is installed.

Two modes:

    python tools/unity_pose_test.py listen            # print incoming packets
    python tools/unity_pose_test.py send              # cycle test poses
    python tools/unity_pose_test.py send --gesture ok # hold one gesture

``listen`` proves the Python side is transmitting.  ``send`` proves a Unity
client is receiving -- and lets you exercise the model without a camera,
MediaPipe or a tracked hand, which is what makes the 3D work testable before
the SolidWorks model exists.

Why the link is worth testing on its own
-----------------------------------------
The pose stream is the one place where a mistake is silent.  A wrong axis order
or a swapped sign does not throw, does not log, and does not stall anything -- it
just drives the wrong part of the model, and you find out by looking at it.  So
the two ends are checked separately, with no vision pipeline in the way.
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.unity_bridge import (  # noqa: E402
    AXIS_ORDER,
    DEFAULT_PORT,
    UnityPoseStream,
)

# Poses to cycle in --send mode.  Values are in the mapper's real output range
# (about 158 extended .. 68 flexed), not the protocol's 0-180, so what you see
# is what a real hand produces.
POSES = {
    "open": {name: 158.0 for name in AXIS_ORDER},
    "half": {name: 115.0 for name in AXIS_ORDER},
    "fist": {name: 68.0 for name in AXIS_ORDER},
    "point": {**{name: 68.0 for name in AXIS_ORDER},
              "index_mcp": 158.0, "index_pip": 165.0},
    "yeah": {**{name: 68.0 for name in AXIS_ORDER},
             "index_mcp": 158.0, "index_pip": 165.0,
             "middle_mcp": 158.0, "middle_pip": 165.0},
}
# A pinch needs the thumb reaching across, and the thumb axes run the other way:
# a low thumb_cmc tucks it in rather than splaying it out.
POSES["pinch"] = {**{name: 68.0 for name in AXIS_ORDER},
                  "index_mcp": 110.0, "index_pip": 100.0,
                  "thumb_cmc": 10.0, "thumb_mcp": 160.0, "thumb_ip": 160.0}


def listen(port: int, host: str) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.settimeout(1.0)

    print(f"listening on udp {host}:{port}  (Ctrl-C to stop)")
    print("waiting for packets...\n")

    count = 0
    started = time.monotonic()
    try:
        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue

            count += 1
            text = data.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                print(f"  [{count}] malformed: {text[:80]!r}")
                continue

            if parsed.get("hello"):
                print(f"hello from {addr[0]}:{addr[1]}")
                print(f"  axis order: {', '.join(parsed.get('n', []))}")
                continue

            angles = parsed.get("a", [])
            summary = " ".join(f"{v:.0f}" for v in angles)
            extra = []
            if parsed.get("g"):
                extra.append(f"gesture={parsed['g']}")
            if parsed.get("o"):
                extra.append(f"orientation={parsed['o']}")
            suffix = ("  " + " ".join(extra)) if extra else ""
            print(f"  [{count:5d}] {summary}{suffix}")

    except KeyboardInterrupt:
        elapsed = time.monotonic() - started
        rate = count / elapsed if elapsed > 0 else 0.0
        print(f"\nreceived {count} packets in {elapsed:.1f}s ({rate:.1f}/s)")
    finally:
        sock.close()
    return 0


def _blend(a: dict, b: dict, t: float) -> dict:
    """Linear interpolation between two poses, by axis name."""
    return {name: a[name] + (b[name] - a[name]) * t for name in a}


def send(port: int, host: str, gesture: str, rate: float, cycles: int,
         hold: float, ramp: float) -> int:
    stream = UnityPoseStream(host=host, port=port)
    if not stream.start():
        print("[FAIL] could not start the stream")
        return 1

    print(f"sending to udp {host}:{port} at {rate:.0f} Hz  (Ctrl-C to stop)")
    print(f"  hold {hold:.1f}s per pose, {ramp:.1f}s to move between poses")

    names = [gesture] if gesture in POSES else list(POSES)
    if gesture and gesture not in POSES:
        print(f"[FAIL] unknown pose {gesture!r}; choose from {', '.join(POSES)}")
        stream.close()
        return 1

    interval = 1.0 / max(1.0, rate)
    sent_cycles = 0
    previous = None

    try:
        while cycles <= 0 or sent_cycles < cycles:
            for name in names:
                target = POSES[name]
                print(f"  -> {name}")

                # Ramp from the previous pose instead of jumping.  A hard cut
                # between two very different poses is unreadable: the model
                # snaps and the eye cannot tell whether the joint moved the
                # right way.  The real pipeline sends continuous angles, so a
                # smooth transition is also closer to what the demo will look
                # like.
                if previous is not None and ramp > 0.0:
                    started = time.monotonic()
                    while True:
                        elapsed = time.monotonic() - started
                        if elapsed >= ramp:
                            break
                        stream.send(_blend(previous, target, elapsed / ramp),
                                    gesture=name)
                        time.sleep(interval)

                started = time.monotonic()
                while time.monotonic() - started < hold:
                    stream.send(target, gesture=name)
                    time.sleep(interval)

                previous = target
            sent_cycles += 1
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()

    print(f"\nsent {stream.packets_sent} packets")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the Unity pose link",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=("listen", "send"), nargs="?", default="listen")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--gesture", default="", help=f"one of: {', '.join(POSES)}")
    parser.add_argument("--rate", type=float, default=60.0, help="packets per second")
    parser.add_argument("--cycles", type=int, default=0, help="0 = loop forever")
    parser.add_argument("--hold", type=float, default=3.0,
                        help="seconds to hold each pose before moving on (default 3)")
    parser.add_argument("--ramp", type=float, default=1.5,
                        help="seconds to interpolate between poses (default 1.5, "
                             "0 = hard cut)")
    parser.add_argument("--slow", action="store_true",
                        help="preset for inspecting joints: 10s hold, 5s ramp")
    args = parser.parse_args()

    if args.slow:
        args.hold, args.ramp = 10.0, 5.0

    if args.mode == "listen":
        return listen(args.port, args.host)
    return send(args.port, args.host, args.gesture, args.rate, args.cycles,
                args.hold, args.ramp)


if __name__ == "__main__":
    raise SystemExit(main())
