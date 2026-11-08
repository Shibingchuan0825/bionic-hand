"""Tests for the hand display's frame-to-frame state.

The display is the only place where a *missing* detection has to be handled
differently from an *unmatched* one: the renderer must keep showing the last
real pose rather than snapping to the rest pose.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_mapper.joint_mapper import JointAngles
from src.ui.display import HandDisplay

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _display() -> HandDisplay:
    # Dashboard only: no camera window, so nothing needs a GUI to exist.
    return HandDisplay(show_camera=False, show_dashboard=True)


def test_a_dropped_detection_keeps_the_last_pose():
    """MediaPipe drops the hand for a frame at a time.

    Rendering the rest pose for those frames made the demo hand snap to
    neutral and back several times a second, which reads as stutter rather
    than as "the hand left the frame".
    """
    display = _display()
    pose = JointAngles()
    pose.index_pip = 40.0

    display.update(camera_frame=FRAME, landmarks=None, angles=pose)
    display.update(camera_frame=FRAME, landmarks=None, angles=None)

    assert display._angles is pose


def test_the_first_frame_without_a_pose_does_not_invent_one():
    display = _display()

    display.update(camera_frame=FRAME, landmarks=None, angles=None)

    assert display._angles is None
