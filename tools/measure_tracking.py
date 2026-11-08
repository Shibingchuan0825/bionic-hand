#!/usr/bin/env python3
"""Measure what the tracking pipeline actually produces.

Answers three questions with numbers instead of guesses:

  * **How fast is it?**  Frames per second, and the time spent per stage.
  * **How far does each joint actually move?**  A fist only bends the joints so
    far; if the mapper's real span is narrow, a renderer calibrated for the
    full protocol range will barely move.
  * **How much does it jitter while held still?**  Peak-to-peak and standard
    deviation per axis tell you whether the tremor is tracking noise or
    rendering.

Run it, move your hand slowly from open to fist a few times, and read the table.

    python tools/measure_tracking.py
    python tools/measure_tracking.py --seconds 20 --camera 1
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hand_tracking.mediapipe_tracker import HandTracker
from src.hand_mapper.joint_mapper import JointMapper
from src.hand_mapper.orientation import OrientationDetector, palm_signed_area

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure tracking pipeline behaviour")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--write-calibration", type=Path, default=None,
        help="Write a Unity sensor-calibration JSON here, e.g. "
             "unity/Assets/Models/sensor_calibration.json")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"[FAIL] model not found: {args.model}")
        return 1

    print("starting tracker (first run pays a 30-60 s XNNPACK warmup)...")
    tracker = HandTracker(model_path=str(args.model), static_image_mode=False)
    # start() returns None and signals failure by raising; treating its return
    # value as a boolean makes every run look like "tracker did not start".
    try:
        tracker.start()
    except Exception as exc:
        print(f"[FAIL] tracker did not start: {type(exc).__name__}: {exc}")
        return 1

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        print(f"[FAIL] could not open camera {args.camera}")
        tracker.stop()
        return 1

    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {args.camera}: {actual_w}x{actual_h}")
    if actual_w == 0 or actual_h == 0:
        print("[FAIL] camera opened but reports no frames; another program may "
              "be holding it")
        capture.release()
        tracker.stop()
        return 1

    mapper = JointMapper()
    orientation_detector = OrientationDetector()
    orientation_counts = {}
    signed_areas = []

    samples = {}
    frame_times = []
    track_times = []
    map_times = []
    detected = 0
    total = 0
    frozen_frames = 0
    previous_frame = None

    print(f"capturing for {args.seconds:.0f}s -- move your hand slowly "
          f"between open and fist\n")

    started = time.perf_counter()
    try:
        while time.perf_counter() - started < args.seconds:
            loop_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                continue
            total += 1

            # A camera that hands back the same image every read looks "alive"
            # (read succeeds, loop runs at 30 fps) while the picture is frozen.
            # Counting identical frames separates that from a real signal.
            if previous_frame is not None and np.array_equal(frame, previous_frame):
                frozen_frames += 1
            previous_frame = frame
            if frozen_frames == 30:
                print("[!] the camera has returned the same frame 30 times -- "
                      "the picture is frozen")

            t0 = time.perf_counter()
            landmarks = tracker.process(frame)
            t1 = time.perf_counter()
            track_times.append((t1 - t0) * 1000.0)

            if landmarks is not None:
                angles = mapper.compute_angles(landmarks)
                t2 = time.perf_counter()
                map_times.append((t2 - t1) * 1000.0)
                orientation = orientation_detector.update(landmarks)
                orientation_counts[orientation.facing] = (
                    orientation_counts.get(orientation.facing, 0) + 1)
                raw_area = palm_signed_area(landmarks)
                if raw_area is not None:
                    signed_areas.append(raw_area)

                if angles is not None:
                    detected += 1
                    for name, value in angles.to_dict().items():
                        samples.setdefault(name, []).append(float(value))

            cv2.putText(frame, f"detected {detected}/{total}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("measure (press q to stop early)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

            frame_times.append((time.perf_counter() - loop_start) * 1000.0)
    finally:
        capture.release()
        tracker.stop()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started

    # Rate over the frames themselves, not over wall time: shutdown and the
    # one-off 30-60 s XNNPACK warmup would otherwise drag it toward 1 fps and
    # make a healthy pipeline look broken.
    busy_ms = sum(frame_times)
    measured_fps = 1000.0 * len(frame_times) / busy_ms if busy_ms > 0 else 0.0

    print(f"\n{'=' * 74}")
    print(f"frames: {total}   detected: {detected} "
          f"({100 * detected / max(total, 1):.0f}%)   wall time: {elapsed:.1f}s")
    print(f"loop rate: {measured_fps:.1f} fps  (measured over frame processing only)")
    if frozen_frames > 10:
        print(f"[!] {frozen_frames} of {total} frames were identical to the "
              f"previous one -- the camera image is frozen")

    def stats(values):
        if not values:
            return "-"
        ordered = sorted(values)
        n = len(ordered)
        return (f"mean {statistics.mean(values):6.1f}  "
                f"p95 {ordered[min(n - 1, int(n * 0.95))]:6.1f}  "
                f"max {ordered[-1]:6.1f}")

    print(f"\nper-stage cost (ms):")
    print(f"  MediaPipe  {stats(track_times)}")
    print(f"  mapping    {stats(map_times)}")
    print(f"  whole loop {stats(frame_times)}")

    print(f"\nper-axis angle span over the capture (degrees):")
    print(f"{'axis':<14} {'min':>7} {'max':>7} {'span':>7} {'std':>7}  bar")
    print("-" * 74)
    narrow = []
    for name in sorted(samples):
        values = samples[name]
        if len(values) < 5:
            continue
        lo, hi = min(values), max(values)
        span = hi - lo
        std = statistics.pstdev(values)
        bar = "#" * min(40, int(span / 2))
        print(f"{name:<14} {lo:>7.1f} {hi:>7.1f} {span:>7.1f} {std:>7.1f}  {bar}")
        if span < 40:
            narrow.append((name, span))

    if args.write_calibration is not None and samples:
        axes = []
        for name in sorted(samples):
            values = samples[name]
            if len(values) < 5:
                continue
            lo, hi = min(values), max(values)
            axes.append({"name": name, "min": round(lo, 2),
                         "max": round(hi, 2), "span": round(hi - lo, 2)})
        document = {
            "source": (f"tools/measure_tracking.py {args.seconds:.0f}s capture, "
                       f"{total} frames, {detected} detections"),
            "axes": axes,
        }
        args.write_calibration.parent.mkdir(parents=True, exist_ok=True)
        args.write_calibration.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + chr(10),
            encoding="utf-8")
        print()
        print(f"[OK] sensor calibration written to {args.write_calibration}")
        print("     HandSceneBuilder reads Assets/Models/sensor_calibration.json;")
        print("     reconstruct the scene in Unity for it to take effect.")

    print()
    print("orientation over the capture:")
    total_orient = sum(orientation_counts.values()) or 1
    for facing in ("front", "back", "side", "unknown"):
        count = orientation_counts.get(facing, 0)
        if count:
            print(f"  {facing:<8} {count:>4} frames ({100 * count / total_orient:>4.0f}%)")
    if signed_areas:
        lo, hi = min(signed_areas), max(signed_areas)
        print(f"  raw palm area: {lo:+.3f} .. {hi:+.3f}  "
              f"(large magnitude = broadside, near zero = edge-on)")
        if abs(hi) < 0.4 and abs(lo) < 0.4:
            print("  [!] the palm never turned far enough broadside to be "
                  "classified confidently -- face the camera and turn the "
                  "hand slowly from palm to back")

    print()
    if not samples:
        print("[!] no hand was detected, so there is nothing to measure.")
        print("    Check the camera view, lighting, and that your whole hand")
        print("    is in frame.")
    elif narrow:
        print(f"[!] {len(narrow)} axis/axes moved less than 40 degrees:")
        for name, span in narrow:
            print(f"      {name}: {span:.0f} deg")
        print("    A renderer calibrated to 68..158 will barely move on these.")
        print("    Either the thresholds need re-calibrating for your hand, or")
        print("    the gesture did not exercise them.")
    else:
        print("[OK] every axis moved at least 40 degrees.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
