from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from time import time
from typing import Sequence
from uuid import uuid4


@dataclass(frozen=True)
class EmbodiedLearningConfig:
    recall_threshold: float = 0.90
    risk_withdrawal_threshold: float = 0.50
    safe_valence_threshold: float = 0.28
    pain_threshold: float = 0.50
    warning_pain_threshold: float = 0.30
    depth_closing_threshold: float = 0.18
    dodge_closing_speed_threshold: float = 10.0
    contact_start_distance: float = 0.90
    contact_full_distance: float = 0.22
    learning_contact_threshold: float = 0.14
    cautious_probe_x: float = 0.20
    close_object_x: float = 1.15
    learning_rate: float = 0.42


@dataclass(frozen=True)
class TactileProfile:
    tip_sharpness: float
    compliance: float
    contact_area: float


@dataclass(frozen=True)
class ContactPoint:
    x: float
    z: float
    sharpness_scale: float = 1.0
    area_scale: float = 1.0


@dataclass(frozen=True)
class SensorPoint:
    x: float
    z: float
    name: str = "sensor"


@dataclass(frozen=True)
class SensorReadings:
    contact: float
    pressure: float
    puncture: float
    pain: float
    safe_touch: bool
    painful: bool
    approaching_sensor: bool
    closing_speed: float
    distance: float
    sensor_name: str = "sensor"


@dataclass
class EmbodiedMemoryEntry:
    label: str
    cue_vector: tuple[float, ...]
    action: str
    valence: float
    risk: float
    strength: float
    pain_peak: float
    pressure_peak: float
    contacts: int
    last_used: float
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class MemoryMatch:
    entry: EmbodiedMemoryEntry
    similarity: float


@dataclass(frozen=True)
class ActionChoice:
    action: str
    reason: str
    confidence: float
    risk: float
    class_name: str
    recall: MemoryMatch | None = None


class EmbodiedMemory:
    def __init__(self, config: EmbodiedLearningConfig | None = None) -> None:
        self.config = config or EmbodiedLearningConfig()
        self.entries: list[EmbodiedMemoryEntry] = []

    def recall(self, cue_vector: Sequence[float]) -> list[MemoryMatch]:
        matches = [
            MemoryMatch(entry=entry, similarity=cosine_similarity(cue_vector, entry.cue_vector))
            for entry in self.entries
        ]
        matches.sort(key=lambda match: match.similarity, reverse=True)
        return matches

    def trusted_recall(self, cue_vector: Sequence[float]) -> MemoryMatch | None:
        matches = self.recall(cue_vector)
        if not matches or matches[0].similarity < self.config.recall_threshold:
            return None
        return matches[0]

    def record_experience(
        self,
        cue_vector: Sequence[float],
        label: str,
        action: str,
        sensors: SensorReadings,
        timestamp: float | None = None,
    ) -> EmbodiedMemoryEntry | None:
        valence, risk, learned = outcome_from_sensors(sensors, self.config)
        if not learned:
            return None

        timestamp = time() if timestamp is None else timestamp
        recalled = self.trusted_recall(cue_vector)
        if recalled is not None:
            entry = recalled.entry
            rate = self.config.learning_rate
            if risk >= self.config.risk_withdrawal_threshold:
                rate = max(rate, 0.78)
            entry.valence = _lerp(entry.valence, valence, rate)
            entry.risk = _clamp01(max(_lerp(entry.risk, risk, rate), entry.risk + max(0.0, risk) * 0.28))
            entry.strength = _clamp01(entry.strength + 0.08 + risk * 0.18)
            entry.pain_peak = max(entry.pain_peak, sensors.pain)
            entry.pressure_peak = max(entry.pressure_peak, sensors.pressure)
            entry.contacts += 1
            entry.action = action
            entry.last_used = timestamp
            return entry

        entry = EmbodiedMemoryEntry(
            label=label,
            cue_vector=tuple(float(value) for value in cue_vector),
            action=action,
            valence=valence,
            risk=risk,
            strength=0.72 if risk > self.config.risk_withdrawal_threshold else 0.48,
            pain_peak=sensors.pain,
            pressure_peak=sensors.pressure,
            contacts=1,
            last_used=timestamp,
        )
        self.entries.append(entry)
        return entry


def choose_action(
    *,
    item_x: float,
    cue_vector: Sequence[float] | None,
    confidence: float,
    sensors: SensorReadings,
    memory: EmbodiedMemory,
    config: EmbodiedLearningConfig | None = None,
) -> ActionChoice:
    config = config or memory.config
    if cue_vector is None:
        return ActionChoice("HOLD", "No object in stereo view.", _clamp01(confidence), 0.0, "unknown")

    current_risk = max(sensors.pain, sensors.puncture)
    if sensors.pain >= config.pain_threshold:
        return ActionChoice(
            "WITHDRAW_FAST",
            "Fingertip pain reflex; storing this contact.",
            _clamp01(confidence),
            _clamp01(current_risk),
            "danger",
        )

    recalled = memory.trusted_recall(cue_vector)
    if recalled and recalled.entry.risk >= config.risk_withdrawal_threshold:
        fast = (
            recalled.entry.pain_peak >= config.pain_threshold
            or item_x > config.close_object_x
            or sensors.approaching_sensor
        )
        reason = (
            "Learned painful cue is closing on the fingertip sensors; maximum avoidance."
            if sensors.approaching_sensor
            else "Vision matched a learned painful cue; maximum avoidance reflex."
        )
        return ActionChoice(
            "WITHDRAW_FAST" if fast else "RETRACT",
            reason,
            _clamp01(confidence),
            _clamp01(recalled.entry.risk),
            "danger",
            recalled,
        )

    if sensors.approaching_sensor and item_x > config.cautious_probe_x:
        return ActionChoice(
            "CAUTIOUS_PROBE",
            "Depth cue shows object closing on the fingertip sensors.",
            _clamp01(confidence),
            _clamp01(current_risk),
            "unknown",
        )

    if recalled and recalled.entry.valence >= config.safe_valence_threshold:
        return ActionChoice(
            "GENTLE_TOUCH" if item_x > config.close_object_x else "APPROACH_SLOW",
            "Learned low-pain contact; reaching cautiously.",
            _clamp01(confidence),
            _clamp01(recalled.entry.risk),
            "safe",
            recalled,
        )

    if item_x > config.cautious_probe_x:
        return ActionChoice(
            "CAUTIOUS_PROBE",
            "No trusted memory yet; probing through fingertip sensors.",
            _clamp01(confidence),
            _clamp01(current_risk),
            "unknown",
        )

    return ActionChoice(
        "ORIENT_CAMERA",
        "Stereo cue seen; waiting for contact evidence.",
        _clamp01(confidence),
        _clamp01(current_risk),
        "unknown",
    )


def should_dodge_reflex(
    choice: ActionChoice,
    sensors: SensorReadings,
    config: EmbodiedLearningConfig | None = None,
) -> bool:
    config = config or EmbodiedLearningConfig()
    return (
        choice.action == "WITHDRAW_FAST"
        and choice.class_name == "danger"
        and choice.recall is not None
        and choice.recall.entry.pain_peak >= config.pain_threshold
        and sensors.closing_speed > config.dodge_closing_speed_threshold
    )


def read_tactile_sensors(
    *,
    sensor_points: Sequence[SensorPoint],
    contact_points: Sequence[ContactPoint],
    tactile_profile: TactileProfile,
    previous_distance: float | None = None,
    delta_seconds: float = 0.0,
    config: EmbodiedLearningConfig | None = None,
) -> SensorReadings:
    config = config or EmbodiedLearningConfig()
    closest_sensor = sensor_points[0] if sensor_points else SensorPoint(0.0, 0.0)
    closest_point = contact_points[0] if contact_points else ContactPoint(0.0, 0.0)
    distance = float("inf")

    for sensor in sensor_points:
        for point in contact_points:
            current = sqrt((sensor.x - point.x) ** 2 + (sensor.z - point.z) ** 2)
            if current < distance:
                distance = current
                closest_sensor = sensor
                closest_point = point

    closing_speed = (
        max(0.0, (previous_distance - distance) / delta_seconds)
        if previous_distance is not None and delta_seconds > 0
        else 0.0
    )
    contact = _clamp01(
        (config.contact_start_distance - distance)
        / (config.contact_start_distance - config.contact_full_distance)
    )
    tip_sharpness = _clamp01(tactile_profile.tip_sharpness * closest_point.sharpness_scale)
    contact_area = _clamp01(tactile_profile.contact_area * closest_point.area_scale)
    pressure = _clamp01(contact * (1.08 - tactile_profile.compliance * 0.35))
    puncture = _clamp01(contact * (0.72 * tip_sharpness + 0.28 * (1.0 - contact_area)))
    pain = _clamp01(pressure * 0.28 + puncture * 0.82 - tactile_profile.compliance * 0.18)

    return SensorReadings(
        contact=contact,
        pressure=pressure,
        puncture=puncture,
        pain=pain,
        safe_touch=contact >= config.learning_contact_threshold and pain < config.pain_threshold,
        painful=pain >= config.pain_threshold,
        approaching_sensor=closing_speed >= config.depth_closing_threshold,
        closing_speed=closing_speed,
        distance=distance,
        sensor_name=closest_sensor.name,
    )


def outcome_from_sensors(
    sensors: SensorReadings, config: EmbodiedLearningConfig | None = None
) -> tuple[float, float, bool]:
    config = config or EmbodiedLearningConfig()
    if sensors.contact < config.learning_contact_threshold:
        return 0.0, 0.0, False

    if sensors.pain >= config.pain_threshold:
        return -max(0.35, sensors.pain), _clamp01(max(0.85, sensors.pain)), True

    if sensors.pain >= config.warning_pain_threshold:
        return (
            -max(0.18, sensors.pain * 0.65),
            _clamp01(max(0.35, sensors.pain + sensors.puncture * 0.35)),
            True,
        )

    comfort = _clamp01(1.0 - sensors.pain - sensors.pressure * 0.18)
    return (
        _clamp01(0.22 + comfort * 0.58),
        _clamp01(sensors.pain * 0.45 + sensors.puncture * 0.20),
        True,
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Cue vector size mismatch: {len(left)} vs {len(right)}")
    dot = sum(l_value * r_value for l_value, r_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    denom = left_norm * right_norm
    return 0.0 if denom == 0 else _clamp(dot / denom, -1.0, 1.0)


def _lerp(left: float, right: float, amount: float) -> float:
    return left + (right - left) * amount


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, float(value)))
