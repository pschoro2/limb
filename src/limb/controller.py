from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from limb.actions import MotorPrimitive


@dataclass(frozen=True)
class MotorCommand:
    action: MotorPrimitive
    accepted: bool
    controller: str
    message: str = ""


class MotorController(Protocol):
    def execute(self, action: MotorPrimitive) -> MotorCommand:
        """Execute or stage a selected motor primitive."""


class DryRunMotorController:
    """Controller boundary for simulation-only runs."""

    def execute(self, action: MotorPrimitive) -> MotorCommand:
        return MotorCommand(
            action=action,
            accepted=True,
            controller="dry_run",
            message="symbolic primitive accepted without robot motion",
        )

