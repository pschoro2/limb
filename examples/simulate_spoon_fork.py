from __future__ import annotations

from limb.actions import MotorPrimitive
from limb.cue import CueEncoder, RobotState
from limb.detector import Detection, StaticDetector
from limb.memory import MemoryBank
from limb.outcome import OutcomeSignals
from limb.policy import SafetySignals, ValenceGatePolicy
from limb.simulation import LimbPipeline
from limb.spikes import SpikeEncoder


VOCAB = (
    "spoon",
    "fork",
    "knife",
    "sharp tool",
    "safe tool",
    "unknown utensil",
    "hand",
    "limb",
    "obstacle",
)


def main() -> None:
    cue_encoder = CueEncoder(VOCAB, embedding_dim=4, limb_pose_dim=2)
    spike_encoder = SpikeEncoder(time_steps=12)
    memory = MemoryBank()
    _seed_known_utensils(memory, cue_encoder, spike_encoder)

    scenarios = [
        ("spoon", _spoon_detection(), SafetySignals(), None),
        ("fork", _fork_detection(), SafetySignals(), None),
        (
            "unknown",
            _unknown_detection(),
            SafetySignals(),
            OutcomeSignals(contact_force=0.2, task_success=False),
        ),
        ("force spike", _spoon_detection(), SafetySignals(force_spike=True), None),
    ]

    for name, detection, safety, outcome in scenarios:
        pipeline = LimbPipeline(
            detector=StaticDetector([detection]),
            cue_encoder=cue_encoder,
            spike_encoder=spike_encoder,
            memory=memory,
            policy=ValenceGatePolicy(),
        )
        step = pipeline.step(
            frame=None,
            robot_state=RobotState(previous_action=MotorPrimitive.HOLD),
            safety_signals=safety,
            outcome=outcome,
        )
        print(f"{name:12s} -> {step.decision.action.value:15s} {step.decision.reason}")


def _seed_known_utensils(memory: MemoryBank, cue_encoder: CueEncoder, spike_encoder: SpikeEncoder) -> None:
    for detection, action, valence, risk in (
        (_spoon_detection(), MotorPrimitive.GENTLE_TOUCH, 1.0, 0.0),
        (_fork_detection(), MotorPrimitive.RETRACT, -1.0, 1.0),
    ):
        cue = cue_encoder.encode(detection, RobotState(previous_action=MotorPrimitive.HOLD))
        memory.record_episode(
            cue_signature=cue.as_memory_vector(),
            path_trace=spike_encoder.encode(cue.vector),
            action_id=action,
            outcome_valence=valence,
            risk_score=risk,
            metadata={"label": detection.label},
        )
        memory.consolidate()


def _spoon_detection() -> Detection:
    return Detection(
        class_id=0,
        label="spoon",
        confidence=0.94,
        bbox_xyxy=(220, 170, 380, 320),
        frame_size=(640, 480),
        class_probs={"spoon": 0.94},
        track_id=10,
        crop_embedding=(0.15, 0.25, 0.1, 0.05),
        depth_m=0.35,
    )


def _fork_detection() -> Detection:
    return Detection(
        class_id=1,
        label="fork",
        confidence=0.89,
        bbox_xyxy=(230, 160, 390, 330),
        frame_size=(640, 480),
        class_probs={"fork": 0.89},
        track_id=20,
        crop_embedding=(0.85, 0.8, 0.75, 0.7),
        depth_m=0.34,
    )


def _unknown_detection() -> Detection:
    return Detection(
        class_id=5,
        label="unknown utensil",
        confidence=0.72,
        bbox_xyxy=(240, 180, 390, 320),
        frame_size=(640, 480),
        class_probs={"unknown utensil": 0.72},
        track_id=30,
        crop_embedding=(0.3, 0.2, 0.5, 0.4),
        depth_m=0.4,
    )


if __name__ == "__main__":
    main()

