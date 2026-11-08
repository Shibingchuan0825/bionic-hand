"""
Joint Angle Mapper.

Maps 21 MediaPipe hand landmarks to servo motor angles for a bionic hand.
Uses vector geometry to compute joint flexion/extension angles.

Joint mapping strategy:
  - MCP joints: Angle between proximal phalanx and metacarpal bone (flexion/extension)
  - PIP joints: Angle between middle and proximal phalanx
  - DIP joints: Angle between distal and middle phalanx
  - Thumb CMC: Angle of thumb metacarpal relative to hand plane
  - Thumb MCP/IP: Flexion angles derived from thumb landmark vectors

Each computed angle is in the range [0, 180] degrees, ready for servo mapping.
"""

from .joint_mapper import JointMapper, JointAngles

__all__ = ["JointMapper", "JointAngles"]
