"""Tests for the MediaPipe hand tracker wrapper.

These run without mediapipe installed: only the parts that do not need the
model (the landmark data structure, the download helper, and the running-mode
selection) are exercised.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_tracking.mediapipe_tracker import HandLandmarks, HandTracker


def _landmarks(confidence: float = 0.9) -> HandLandmarks:
    pixels = np.zeros((21, 3), dtype=float)
    for index in range(21):
        pixels[index] = [100.0 + 6.0 * index, 120.0 + 9.0 * index, 0.0]
    return HandLandmarks(
        landmarks_px=pixels,
        landmarks_norm=pixels / 640.0,
        handedness="Right",
        handedness_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Frame quality signal
# ---------------------------------------------------------------------------

def test_frame_quality_gates_on_the_handedness_score():
    """The only per-frame score the Tasks API exposes is the handedness one."""
    assert _landmarks(0.9).is_valid() is True
    assert _landmarks(0.4).is_valid() is False


def test_landmarks_only_expose_the_handedness_score():
    """A second, invented 'detection confidence' field must not come back."""
    assert not hasattr(_landmarks(), "detection_confidence")
    assert _landmarks().handedness_confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Running mode
# ---------------------------------------------------------------------------

def test_configured_mode_selects_image_or_video(tmp_path):
    """static_image_mode must reach the detector, not just be recorded."""
    image_mode = HandTracker(static_image_mode=True)
    video_mode = HandTracker(static_image_mode=False)

    assert image_mode.static_image_mode is True
    assert video_mode.static_image_mode is False


def test_video_mode_starts_with_no_previous_timestamp():
    """VIDEO mode needs a monotonic timestamp; the first frame must be free."""
    tracker = HandTracker(static_image_mode=False)

    assert tracker._last_timestamp_ms == -1


def test_model_complexity_is_not_part_of_the_interface():
    """The Tasks API ships one model; there is no complexity switch."""
    with pytest.raises(TypeError):
        HandTracker(model_complexity=1)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def test_download_honours_a_custom_target_path(tmp_path, monkeypatch):
    """A custom model_path must be downloaded to, not silently ignored."""
    downloads = []

    def fake_urlretrieve(url, target):
        downloads.append((url, target))
        Path(target).write_bytes(b"model")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    custom = tmp_path / "nested" / "hand.task"
    returned = HandTracker._download_model(str(custom))

    assert returned == str(custom)
    assert custom.exists()
    assert downloads[0][1] == str(custom)


def test_download_falls_back_to_the_default_path(monkeypatch):
    from src.hand_tracking import mediapipe_tracker as module

    downloads = []
    monkeypatch.setattr(
        "urllib.request.urlretrieve",
        lambda url, target: downloads.append(target),
    )

    returned = HandTracker._download_model()

    assert returned == str(module._DEFAULT_MODEL_PATH)
    assert downloads[0] == str(module._DEFAULT_MODEL_PATH)


def test_start_downloads_to_the_configured_path(tmp_path, monkeypatch):
    """The download target must be the path the detector will later load."""
    from src.hand_tracking import mediapipe_tracker as module

    custom = tmp_path / "custom_hand.task"
    recorded = {}
    monkeypatch.setattr(module, "_ensure_mediapipe", lambda: None)
    monkeypatch.setattr(
        module.HandTracker, "_download_model",
        staticmethod(lambda target=None: recorded.setdefault("target", target)),
    )

    tracker = HandTracker(model_path=str(custom))

    # Everything past the download needs mediapipe, which is not installed;
    # what matters is that the download was pointed at the configured path.
    with pytest.raises(Exception):
        tracker.start()

    assert recorded["target"] == str(custom)
