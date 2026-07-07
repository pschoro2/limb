from __future__ import annotations

from enum import Enum


class MotorPrimitive(str, Enum):
    HOLD = "HOLD"
    ORIENT_CAMERA = "ORIENT_CAMERA"
    RETRACT = "RETRACT"
    WITHDRAW_FAST = "WITHDRAW_FAST"
    CAUTIOUS_PROBE = "CAUTIOUS_PROBE"
    APPROACH_SLOW = "APPROACH_SLOW"
    GENTLE_TOUCH = "GENTLE_TOUCH"
    GRASP_LIGHT = "GRASP_LIGHT"
    SUPPRESS_APPROACH = "SUPPRESS_APPROACH"


APPROACH_ACTIONS = {
    MotorPrimitive.APPROACH_SLOW,
    MotorPrimitive.CAUTIOUS_PROBE,
    MotorPrimitive.GENTLE_TOUCH,
    MotorPrimitive.GRASP_LIGHT,
}

