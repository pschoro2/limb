from __future__ import annotations

import pytest

from limb.actions import MotorPrimitive
from limb.cue import CueEncoder, RobotState
from limb.detector import Detection, StaticDetector, resolve_local_model_path
from limb.memory import MemoryBank, MemoryEntry
from limb.outcome import OutcomeSignals, compute_valence
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


def test_spoon_memory_selects_gentle_touch() -> None:
    pipeline = _pipeline_with_seeded_memories(_spoon_detection())

    step = pipeline.step(frame=None, robot_state=RobotState(previous_action=MotorPrimitive.HOLD))

    assert step.decision.action == MotorPrimitive.GENTLE_TOUCH
    assert step.decision.reason == "positive_memory_allows_action"


def test_fork_memory_retracts_from_negative_valence() -> None:
    pipeline = _pipeline_with_seeded_memories(_fork_detection())

    step = pipeline.step(frame=None, robot_state=RobotState(previous_action=MotorPrimitive.HOLD))

    assert step.decision.action == MotorPrimitive.RETRACT
    assert step.decision.reason == "negative_valence_or_high_risk"


def test_low_confidence_unknown_holds() -> None:
    pipeline = _pipeline_with_seeded_memories(
        Detection(
            class_id=5,
            label="unknown utensil",
            confidence=0.41,
            bbox_xyxy=(240, 180, 390, 320),
            frame_size=(640, 480),
            class_probs={"unknown utensil": 0.41},
            crop_embedding=(0.3, 0.2, 0.5),
            depth_m=0.4,
        )
    )

    step = pipeline.step(frame=None, robot_state=RobotState(previous_action=MotorPrimitive.HOLD))

    assert step.decision.action == MotorPrimitive.HOLD
    assert step.decision.reason == "low_yolo_confidence"


def test_novel_but_confident_unknown_gets_cautious_probe() -> None:
    pipeline = _pipeline_with_seeded_memories(
        Detection(
            class_id=5,
            label="unknown utensil",
            confidence=0.72,
            bbox_xyxy=(240, 180, 390, 320),
            frame_size=(640, 480),
            class_probs={"unknown utensil": 0.72},
            crop_embedding=(0.3, 0.2, 0.5),
            depth_m=0.4,
        )
    )

    step = pipeline.step(frame=None, robot_state=RobotState(previous_action=MotorPrimitive.HOLD))

    assert step.decision.action == MotorPrimitive.CAUTIOUS_PROBE
    assert step.decision.reason == "novel_object_probe"


def test_hard_safety_overrides_positive_memory() -> None:
    pipeline = _pipeline_with_seeded_memories(_spoon_detection())

    step = pipeline.step(
        frame=None,
        robot_state=RobotState(previous_action=MotorPrimitive.HOLD),
        safety_signals=SafetySignals(force_spike=True),
    )

    assert step.decision.action == MotorPrimitive.WITHDRAW_FAST
    assert step.decision.reason == "hard_safety_withdraw"
    assert step.motor_command.action == MotorPrimitive.WITHDRAW_FAST
    assert step.motor_command.controller == "dry_run"


def test_outcome_valence_penalizes_danger_signals() -> None:
    valence = compute_valence(
        OutcomeSignals(
            contact_force=2.0,
            safe_force_limit=1.0,
            task_success=False,
            force_spike=True,
            collision_detected=True,
            emergency_stop=True,
        )
    )

    assert valence == -2.0


def test_yolo_model_resolution_is_local_only(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="downloads are disabled"):
        resolve_local_model_path()

    local_model = tmp_path / "yolo5s.pt"
    local_model.write_bytes(b"local test model placeholder")

    assert resolve_local_model_path() == local_model.relative_to(tmp_path)


def test_consolidation_moves_promoted_flash_memories() -> None:
    cue_encoder = CueEncoder(VOCAB, embedding_dim=4, limb_pose_dim=2)
    spike_encoder = SpikeEncoder(time_steps=12)
    memory = MemoryBank()
    cue = cue_encoder.encode(_fork_detection(), RobotState(previous_action=MotorPrimitive.HOLD))

    memory.record_episode(
        cue_signature=cue.as_memory_vector(),
        path_trace=spike_encoder.encode(cue.vector),
        action_id=MotorPrimitive.RETRACT,
        outcome_valence=-1.0,
        risk_score=1.0,
        metadata={"label": "fork"},
    )
    promoted = memory.consolidate()

    assert len(promoted) == 1
    assert memory.flash == []
    assert len(memory.consolidated) == 1


def _pipeline_with_seeded_memories(current_detection: Detection) -> LimbPipeline:
    cue_encoder = CueEncoder(VOCAB, embedding_dim=4, limb_pose_dim=2)
    spike_encoder = SpikeEncoder(time_steps=12)
    memory = MemoryBank()
    _seed_memory(memory, cue_encoder, spike_encoder, _spoon_detection(), MotorPrimitive.GENTLE_TOUCH, 1.0, 0.0)
    _seed_memory(memory, cue_encoder, spike_encoder, _fork_detection(), MotorPrimitive.RETRACT, -1.0, 1.0)
    return LimbPipeline(
        detector=StaticDetector([current_detection]),
        cue_encoder=cue_encoder,
        spike_encoder=spike_encoder,
        memory=memory,
        policy=ValenceGatePolicy(),
    )


def _seed_memory(
    memory: MemoryBank,
    cue_encoder: CueEncoder,
    spike_encoder: SpikeEncoder,
    detection: Detection,
    action: MotorPrimitive,
    valence: float,
    risk_score: float,
) -> None:
    cue = cue_encoder.encode(detection, RobotState(previous_action=MotorPrimitive.HOLD))
    trace = spike_encoder.encode(cue.vector)
    memory.add(
        MemoryEntry(
            cue_signature=cue.as_memory_vector(),
            path_trace=trace,
            action_id=action,
            outcome_valence=valence,
            risk_score=risk_score,
            strength=0.9,
            source="consolidated",
            metadata={"label": detection.label},
        )
    )


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
