from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from limb.actions import MotorPrimitive
from limb.detector import Detection


@dataclass(frozen=True)
class RobotState:
    limb_pose: Sequence[float] = field(default_factory=tuple)
    previous_action: MotorPrimitive = MotorPrimitive.HOLD


@dataclass(frozen=True)
class CueSignature:
    vector: np.ndarray
    feature_names: tuple[str, ...]
    detection: Detection

    def as_memory_vector(self) -> np.ndarray:
        return self.vector.astype(float, copy=True)


class CueEncoder:
    """Builds a cue signature from YOLO output plus robot context."""

    def __init__(
        self,
        class_vocab: Sequence[str],
        embedding_dim: int = 16,
        limb_pose_dim: int = 6,
        max_depth_m: float = 2.0,
        motion_scale: float = 0.25,
        action_vocab: Sequence[MotorPrimitive] | None = None,
        class_weight: float = 3.0,
        confidence_weight: float = 0.75,
        geometry_weight: float = 1.0,
        depth_weight: float = 0.75,
        motion_weight: float = 0.5,
        embedding_weight: float = 2.0,
        limb_pose_weight: float = 0.5,
        previous_action_weight: float = 0.25,
    ) -> None:
        if not class_vocab:
            raise ValueError("class_vocab must not be empty")
        self.class_vocab = tuple(class_vocab)
        self.embedding_dim = embedding_dim
        self.limb_pose_dim = limb_pose_dim
        self.max_depth_m = max_depth_m
        self.motion_scale = motion_scale
        self.action_vocab = tuple(action_vocab or MotorPrimitive)
        self.class_weight = class_weight
        self.confidence_weight = confidence_weight
        self.geometry_weight = geometry_weight
        self.depth_weight = depth_weight
        self.motion_weight = motion_weight
        self.embedding_weight = embedding_weight
        self.limb_pose_weight = limb_pose_weight
        self.previous_action_weight = previous_action_weight
        self._last_centers: dict[int, tuple[float, float]] = {}

    def encode(self, detection: Detection, robot_state: RobotState | None = None) -> CueSignature:
        robot_state = robot_state or RobotState()
        class_features = self._class_probability_vector(detection)
        confidence = np.array([_clamp01(detection.confidence)], dtype=float)
        geometry = np.array(detection.normalized_center_size, dtype=float)
        depth = np.array([self._normalized_depth(detection.depth_m)], dtype=float)
        motion = np.array(self._motion_features(detection), dtype=float)
        embedding = np.array(_pad_or_trim(detection.crop_embedding, self.embedding_dim), dtype=float)
        limb_pose = np.array(_pad_or_trim(robot_state.limb_pose, self.limb_pose_dim), dtype=float)
        limb_pose = np.clip(limb_pose, -1.0, 1.0)
        previous_action = self._previous_action_vector(robot_state.previous_action)

        vector = np.concatenate(
            [
                class_features * self.class_weight,
                confidence * self.confidence_weight,
                geometry * self.geometry_weight,
                depth * self.depth_weight,
                motion * self.motion_weight,
                embedding * self.embedding_weight,
                limb_pose * self.limb_pose_weight,
                previous_action * self.previous_action_weight,
            ]
        )
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)

        names = (
            tuple(f"class:{name}" for name in self.class_vocab)
            + ("det_confidence",)
            + ("bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height")
            + ("depth",)
            + ("motion_dx", "motion_dy")
            + tuple(f"embedding:{i}" for i in range(self.embedding_dim))
            + tuple(f"limb_pose:{i}" for i in range(self.limb_pose_dim))
            + tuple(f"previous_action:{action.value}" for action in self.action_vocab)
        )
        return CueSignature(vector=vector, feature_names=names, detection=detection)

    def _class_probability_vector(self, detection: Detection) -> np.ndarray:
        explicit_probs = _normalize_mapping(detection.class_probs)
        values = []
        for label in self.class_vocab:
            if label in explicit_probs:
                values.append(explicit_probs[label])
            elif label == detection.label:
                values.append(_clamp01(detection.confidence))
            else:
                values.append(0.0)
        return np.array(values, dtype=float)

    def _normalized_depth(self, depth_m: float | None) -> float:
        if depth_m is None:
            return 0.0
        if self.max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive")
        return _clamp01(depth_m / self.max_depth_m)

    def _motion_features(self, detection: Detection) -> tuple[float, float]:
        center_x, center_y, _, _ = detection.normalized_center_size
        if detection.track_id is None:
            return (0.0, 0.0)
        previous = self._last_centers.get(detection.track_id)
        self._last_centers[detection.track_id] = (center_x, center_y)
        if previous is None:
            return (0.0, 0.0)
        dx = (center_x - previous[0]) / self.motion_scale
        dy = (center_y - previous[1]) / self.motion_scale
        return (_clamp_signed(dx), _clamp_signed(dy))

    def _previous_action_vector(self, previous_action: MotorPrimitive) -> np.ndarray:
        return np.array([1.0 if action == previous_action else 0.0 for action in self.action_vocab], dtype=float)


def _normalize_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {str(key): _clamp01(float(value)) for key, value in values.items()}


def _pad_or_trim(values: Sequence[float], target_dim: int) -> list[float]:
    if target_dim < 0:
        raise ValueError("target_dim must be non-negative")
    vector = [float(value) for value in values[:target_dim]]
    if len(vector) < target_dim:
        vector.extend([0.0] * (target_dim - len(vector)))
    return vector


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _clamp_signed(value: float) -> float:
    return min(1.0, max(-1.0, float(value)))
