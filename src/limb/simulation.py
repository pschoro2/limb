from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from limb.controller import DryRunMotorController, MotorCommand, MotorController
from limb.cue import CueEncoder, CueSignature, RobotState
from limb.detector import Detection, Detector
from limb.memory import MemoryBank, RetrievedMemory
from limb.outcome import OutcomeSignals, compute_valence, risk_from_valence
from limb.policy import ActionDecision, SafetySignals, ValenceGatePolicy
from limb.spikes import SpikeEncoder, SpikeTrace


@dataclass(frozen=True)
class PipelineStep:
    detections: tuple[Detection, ...]
    selected_detection: Detection | None
    cue: CueSignature | None
    spike_trace: SpikeTrace | None
    retrieved: RetrievedMemory
    decision: ActionDecision
    motor_command: MotorCommand
    outcome_valence: float | None = None


class LimbPipeline:
    def __init__(
        self,
        detector: Detector,
        cue_encoder: CueEncoder,
        spike_encoder: SpikeEncoder,
        memory: MemoryBank,
        policy: ValenceGatePolicy,
        motor_controller: MotorController | None = None,
    ) -> None:
        self.detector = detector
        self.cue_encoder = cue_encoder
        self.spike_encoder = spike_encoder
        self.memory = memory
        self.policy = policy
        self.motor_controller = motor_controller or DryRunMotorController()

    def step(
        self,
        frame: Any,
        robot_state: RobotState | None = None,
        safety_signals: SafetySignals | None = None,
        outcome: OutcomeSignals | None = None,
        learn_from_outcome: bool = True,
    ) -> PipelineStep:
        detections = tuple(self.detector.predict(frame))
        selected = _highest_confidence(detections)

        if selected is None:
            retrieved = RetrievedMemory(matches=tuple())
            decision = self.policy.select(None, retrieved, safety_signals=safety_signals)
            motor_command = self.motor_controller.execute(decision.action)
            return PipelineStep(
                detections=detections,
                selected_detection=None,
                cue=None,
                spike_trace=None,
                retrieved=retrieved,
                decision=decision,
                motor_command=motor_command,
            )

        cue = self.cue_encoder.encode(selected, robot_state)
        spike_trace = self.spike_encoder.encode(cue.vector)
        retrieved = self.memory.retrieve(cue.vector)
        decision = self.policy.select(selected, retrieved, safety_signals=safety_signals)
        motor_command = self.motor_controller.execute(decision.action)

        outcome_valence = None
        if outcome is not None:
            outcome_valence = compute_valence(outcome)
            if learn_from_outcome:
                self.memory.record_episode(
                    cue_signature=cue.as_memory_vector(),
                    path_trace=spike_trace,
                    action_id=decision.action,
                    outcome_valence=outcome_valence,
                    risk_score=risk_from_valence(outcome_valence),
                    metadata={"label": selected.label, "reason": decision.reason},
                )

        return PipelineStep(
            detections=detections,
            selected_detection=selected,
            cue=cue,
            spike_trace=spike_trace,
            retrieved=retrieved,
            decision=decision,
            motor_command=motor_command,
            outcome_valence=outcome_valence,
        )


def _highest_confidence(detections: tuple[Detection, ...]) -> Detection | None:
    if not detections:
        return None
    return max(detections, key=lambda detection: detection.confidence)
