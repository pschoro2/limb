from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutcomeSignals:
    contact_force: float = 0.0
    safe_force_limit: float = 1.0
    task_success: bool = False
    force_spike: bool = False
    collision_detected: bool = False
    emergency_stop: bool = False
    human_override: bool = False
    unexpected_torque: bool = False
    sharp_contact: bool = False
    slip: bool = False


def compute_valence(outcome: OutcomeSignals) -> float:
    valence = 0.0

    if outcome.contact_force <= outcome.safe_force_limit:
        valence += 0.4
    if outcome.task_success:
        valence += 0.6
    if outcome.slip:
        valence -= 0.2
    if outcome.force_spike:
        valence -= 1.0
    if outcome.collision_detected:
        valence -= 1.0
    if outcome.sharp_contact:
        valence -= 1.0
    if outcome.unexpected_torque:
        valence -= 0.8
    if outcome.human_override:
        valence -= 1.0
    if outcome.emergency_stop:
        valence -= 2.0

    return max(-2.0, min(1.0, valence))


def risk_from_valence(valence: float) -> float:
    return max(0.0, min(1.0, -float(valence)))

