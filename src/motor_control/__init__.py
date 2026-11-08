"""
Motor Control Module.

Provides servo management and serial communication with the bionic hand hardware.
"""
from .serial_controller import SerialController, RealSerial, SimulatedSerial, SerialInterface
from .servo_manager import ServoManager, ServoConfig, ServoState, ServoStateInfo

__all__ = [
    "SerialController",
    "RealSerial",
    "SimulatedSerial",
    "SerialInterface",
    "ServoManager",
    "ServoConfig",
    "ServoState",
    "ServoStateInfo",
]
