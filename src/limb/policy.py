from __future__ import annotations

from dataclasses import dataclass

from limb.actions import APPROACH_ACTIONS, MotorPrimitive
from limb.detector import Detection
from limb.memory import MemoryMatch, RetrievedMemory


@dataclass(frozen=True)
class SafetySignals:
    emergency_stop: bool = False
    force_spike: bool = False
    collision: bool = False
    high_force: bool = False
    human_stop: bool = False
    sensor_fault: bool = False


@dataclass(frozen=True)
class PolicyConfig:
    theta_safe: float = 0.52
    theta_recall: float = 0.78
    theta_margin: float = 0.06
    theta_probe: float = 0.55
    min_detection_confidence: float = 0.45
    danger_threshold: float = -0.2
    safe_threshold: float = 0.2
    high_risk_threshold: float = 0.7


@dataclass(frozen=True)
class ConfidenceEstimate:
    overall: float
    yolo_confidence: float
    memory_similarity: float
    margin: float
    memory_strength: float
    valence_certainty: float


@dataclass(frozen=True)
class ActionDecision:
    action: MotorPrimitive
    reason: str
    confidence: ConfidenceEstimate
    match: MemoryMatch | None = None


class SafetySupervisor:
    """Fixed safety layer outside learned memory/action selection."""

    def override(self, signals: SafetySignals) -> tuple[MotorPrimitive, str] | None:
        if signals.emergency_stop or signals.force_spike or signals.collision or signals.high_force or signals.human_stop:
            return (MotorPrimitive.WITHDRAW_FAST, "hard_safety_withdraw")
        if signals.sensor_fault:
            return (MotorPrimitive.HOLD, "sensor_fault_hold")
        return None


class ValenceGatePolicy:
    def __init__(
        self,
        config: PolicyConfig | None = None,
        safety_supervisor: SafetySupervisor | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self.safety_supervisor = safety_supervisor or SafetySupervisor()

    def select(
        self,
        detection: Detection | None,
        retrieved: RetrievedMemory,
        safety_signals: SafetySignals | None = None,
    ) -> ActionDecision:
        safety_signals = safety_signals or SafetySignals()
        confidence = self._estimate_confidence(detection, retrieved.best, retrieved.margin)
        override = self.safety_supervisor.override(safety_signals)
        if override is not None:
            action, reason = override
            return ActionDecision(action=action, reason=reason, confidence=confidence, match=retrieved.best)

        if detection is None:
            return ActionDecision(
                action=MotorPrimitive.HOLD,
                reason="no_detection",
                confidence=confidence,
                match=None,
            )

        if detection.confidence < self.config.min_detection_confidence:
            return ActionDecision(
                action=MotorPrimitive.HOLD,
                reason="low_yolo_confidence",
                confidence=confidence,
                match=retrieved.best,
            )

        best = retrieved.best
        if best is None or best.similarity < self.config.theta_recall:
            if detection.confidence >= self.config.theta_probe:
                return ActionDecision(
                    action=MotorPrimitive.CAUTIOUS_PROBE,
                    reason="novel_object_probe",
                    confidence=confidence,
                    match=best,
                )
            return ActionDecision(
                action=MotorPrimitive.HOLD,
                reason="novel_object_hold",
                confidence=confidence,
                match=best,
            )

        if retrieved.second is not None and retrieved.margin < self.config.theta_margin:
            return ActionDecision(
                action=MotorPrimitive.ORIENT_CAMERA,
                reason="ambiguous_memory_second_view",
                confidence=confidence,
                match=best,
            )

        if confidence.overall < self.config.theta_safe:
            return ActionDecision(
                action=MotorPrimitive.HOLD,
                reason="low_combined_confidence",
                confidence=confidence,
                match=best,
            )

        if best.entry.risk_score >= self.config.high_risk_threshold or best.entry.outcome_valence < self.config.danger_threshold:
            return ActionDecision(
                action=MotorPrimitive.RETRACT,
                reason="negative_valence_or_high_risk",
                confidence=confidence,
                match=best,
            )

        if best.entry.outcome_valence > self.config.safe_threshold:
            action = best.entry.action_id
            if action in APPROACH_ACTIONS:
                return ActionDecision(
                    action=action,
                    reason="positive_memory_allows_action",
                    confidence=confidence,
                    match=best,
                )
            return ActionDecision(
                action=action,
                reason="positive_memory_nonapproach_action",
                confidence=confidence,
                match=best,
            )

        return ActionDecision(
            action=MotorPrimitive.HOLD,
            reason="uncertain_valence_hold",
            confidence=confidence,
            match=best,
        )

    def _estimate_confidence(
        self,
        detection: Detection | None,
        best: MemoryMatch | None,
        margin: float,
    ) -> ConfidenceEstimate:
        yolo_confidence = detection.confidence if detection is not None else 0.0
        memory_similarity = best.similarity if best is not None else 0.0
        memory_strength = best.entry.strength if best is not None else 0.0
        valence_certainty = min(1.0, abs(best.entry.outcome_valence)) if best is not None else 0.0
        capped_margin = min(1.0, max(0.0, margin / max(self.config.theta_margin, 1e-6)))
        overall = (
            0.30 * yolo_confidence
            + 0.30 * memory_similarity
            + 0.15 * capped_margin
            + 0.15 * memory_strength
            + 0.10 * valence_certainty
        )
        return ConfidenceEstimate(
            overall=min(1.0, max(0.0, overall)),
            yolo_confidence=yolo_confidence,
            memory_similarity=memory_similarity,
            margin=margin,
            memory_strength=memory_strength,
            valence_certainty=valence_certainty,
        )

