#!/usr/bin/env python
"""Measure how many frames it takes to recognise a pose change.

Builds a synthetic 21-landmark hand whose curl is a single parameter, feeds it
through the real JointMapper and GestureRecognizer, and reports the delay
between "the hand started to close" and "the recogniser said fist".

That delay is the sum of every lag in the chain -- max_step, the EMA, the
dead band and the hold-frame debounce -- so it is the honest number to quote
when someone says recognition feels slow.

Usage:
    python tools/measure_gesture_latency.py
    python tools/measure_gesture_latency.py --fps 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_mapper.gestures import GestureRecognizer
from src.hand_mapper.joint_mapper import JointMapper
from src.hand_tracking.mediapipe_tracker import HandLandmarks

# Proximal : middle : distal = 1 : 0.67 : 0.5, the RUKA v2 ratio.
PROXIMAL, MIDDLE, DISTAL = 45.0, 30.0, 22.5

# Finger -> (MCP x, MCP y).  Image coordinates: y grows downwards.
MCP = {
    5:  (280.0, 380.0),
    9:  (310.0, 375.0),
    13: (340.0, 378.0),
    17: (370.0, 385.0),
}

WRIST = (320.0, 500.0)

# Joint bend at full curl, degrees.
MCP_BEND, PIP_BEND, DIP_BEND = 70.0, 90.0, 45.0

# Thumb segments and fold angles.  The fold has to carry thumb_ip below
# EXTENDED_MIN so a closed hand stops matching thumbs_up.
THUMB_PROXIMAL, THUMB_MIDDLE, THUMB_DISTAL = 32.0, 26.0, 22.0
THUMB_MCP_BEND, THUMB_IP_BEND = 45.0, 75.0


def _dir(angle_deg):
    r = np.radians(angle_deg)
    return np.array([np.cos(r), np.sin(r), 0.0])


def build_hand(curl: float) -> HandLandmarks:
    """Return landmarks for a hand curled by ``curl`` (0 open .. 1 fist)."""
    px = np.zeros((21, 3), dtype=float)
    px[0] = [WRIST[0], WRIST[1], 0.0]

    for mcp_idx, (mx, my) in MCP.items():
        base = -90.0                       # straight up in image coords
        a1 = base + MCP_BEND * curl
        a2 = a1 + PIP_BEND * curl
        a3 = a2 + DIP_BEND * curl
        pip = np.array([mx, my, 0.0]) + _dir(a1) * PROXIMAL
        dip = pip + _dir(a2) * MIDDLE
        tip = dip + _dir(a3) * DISTAL
        px[mcp_idx] = [mx, my, 0.0]
        px[mcp_idx + 1] = pip
        px[mcp_idx + 2] = dip
        px[mcp_idx + 3] = tip

    # Thumb: CMC, MCP, IP, TIP.  Same parametric construction as a finger so
    # that it actually folds -- a thumb left straight makes every closed pose
    # match thumbs_up, which sits before fist in the gesture table.
    t0 = -115.0
    t1 = t0 + THUMB_MCP_BEND * curl
    t2 = t1 + THUMB_IP_BEND * curl
    px[1] = [258.0, 430.0, 0.0]
    px[2] = np.array([258.0, 430.0, 0.0]) + _dir(t0) * THUMB_PROXIMAL
    px[3] = px[2] + _dir(t1) * THUMB_MIDDLE
    px[4] = px[3] + _dir(t2) * THUMB_DISTAL

    return HandLandmarks(
        landmarks_px=px,
        landmarks_norm=px / 640.0,
        handedness="Right",
        handedness_confidence=0.95,
    )


def _run(use_instant, fps, settle, trace):
    """Feed settle+close frames; return (frame the fist was reported, trace rows)."""
    mapper = JointMapper()
    recognizer = GestureRecognizer()
    rows = []

    for _ in range(settle):
        hand = build_hand(0.0)
        angles = mapper.compute_angles(hand)
        recognizer.update(mapper.instant_angles if use_instant else angles, hand)

    flip = None
    for i in range(1, trace + 1):
        hand = build_hand(1.0)
        angles = mapper.compute_angles(hand)
        fed = mapper.instant_angles if use_instant else angles
        result = recognizer.update(fed, hand)
        if result.name == "fist" and flip is None:
            flip = i
        rows.append((i, float(angles.index_pip), float(fed.index_pip),
                     result.name))
    return flip, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--settle", type=int, default=20)
    parser.add_argument("--trace", type=int, default=40)
    args = parser.parse_args()

    print("Synthetic hand closes from open to fist in one frame.")
    print("Reporting the frame on which the recogniser first says 'fist'.")
    print()

    results = {}
    for label, use_instant in (("smoothed (old)", False), ("instant  (new)", True)):
        flip, rows = _run(use_instant, args.fps, args.settle, args.trace)
        results[label] = (flip, rows)

    print(f"{'frame':>5} | {'smoothed':>9} {'old':>6} | {'instant':>8} {'new':>6}")
    print("-" * 46)
    for (i, sm, _, old_name), (_, _, inst, new_name) in zip(
            results["smoothed (old)"][1], results["instant  (new)"][1]):
        print(f"{i:>5} | {sm:>9.1f} {old_name or '-':>6} | {inst:>8.1f} {new_name or '-':>6}")

    print()
    for label, (flip, _) in results.items():
        if flip is None:
            print(f"{label}: not recognised within {args.trace} frames")
        else:
            print(f"{label}: frame {flip:>3}  = {flip / args.fps * 1000:>6.0f} ms "
                  f"at {args.fps:.0f} fps")


if __name__ == "__main__":
    main()
