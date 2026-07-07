from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


EmbeddingFn = Callable[[Any, tuple[float, float, float, float]], Sequence[float]]
LOCAL_MODEL_CANDIDATES = ("yolo5s.pt", "yolov5s.pt")


@dataclass(frozen=True)
class Detection:
    """Detector output normalized into the fields used by the memory system."""

    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_size: tuple[int, int]
    class_probs: Mapping[str, float] = field(default_factory=dict)
    track_id: int | None = None
    mask: Any | None = None
    crop_embedding: Sequence[float] = field(default_factory=tuple)
    depth_m: float | None = None

    @property
    def normalized_bbox(self) -> tuple[float, float, float, float]:
        width, height = self.frame_size
        x1, y1, x2, y2 = self.bbox_xyxy
        if width <= 0 or height <= 0:
            raise ValueError("frame_size must contain positive width and height")
        return (
            _clamp01(x1 / width),
            _clamp01(y1 / height),
            _clamp01(x2 / width),
            _clamp01(y2 / height),
        )

    @property
    def normalized_center_size(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.normalized_bbox
        bbox_w = max(0.0, x2 - x1)
        bbox_h = max(0.0, y2 - y1)
        return (
            _clamp01(x1 + bbox_w / 2.0),
            _clamp01(y1 + bbox_h / 2.0),
            _clamp01(bbox_w),
            _clamp01(bbox_h),
        )


class Detector(Protocol):
    def predict(self, frame: Any) -> list[Detection]:
        """Return object detections for a frame."""


class StaticDetector:
    """Small detector useful for simulations and deterministic tests."""

    def __init__(self, detections: Sequence[Detection]):
        self._detections = list(detections)

    def predict(self, frame: Any) -> list[Detection]:
        return list(self._detections)


class UltralyticsYOLODetector:
    """Adapter for Ultralytics YOLO detection/tracking outputs.

    The import is intentionally lazy so the rest of the package can run without
    installing the optional YOLO dependency.
    """

    def __init__(
        self,
        model_name: str | Path | None = None,
        confidence: float = 0.4,
        crop_embedding_fn: EmbeddingFn | None = None,
        tracker: bool = False,
        require_local_model: bool = True,
    ) -> None:
        model_path = resolve_local_model_path(model_name, require_local_model=require_local_model)

        from ultralytics import YOLO

        self.model_path = model_path
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.crop_embedding_fn = crop_embedding_fn
        self.tracker = tracker

    def predict(self, frame: Any) -> list[Detection]:
        if self.tracker:
            results = self.model.track(frame, conf=self.confidence, persist=True, verbose=False)
        else:
            results = self.model.predict(frame, conf=self.confidence, verbose=False)

        detections: list[Detection] = []
        for result in results:
            frame_height, frame_width = _result_shape(result, frame)
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                label = str(names.get(class_id, class_id))
                confidence = float(box.conf[0])
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                track_id = _box_track_id(box)
                embedding = (
                    tuple(float(v) for v in self.crop_embedding_fn(frame, xyxy))
                    if self.crop_embedding_fn
                    else tuple()
                )

                detections.append(
                    Detection(
                        class_id=class_id,
                        label=label,
                        confidence=confidence,
                        bbox_xyxy=xyxy,  # type: ignore[arg-type]
                        frame_size=(frame_width, frame_height),
                        class_probs={label: confidence},
                        track_id=track_id,
                        crop_embedding=embedding,
                    )
                )

        return detections


def resolve_local_model_path(
    model_name: str | Path | None = None,
    *,
    require_local_model: bool = True,
) -> Path:
    if model_name is None:
        for candidate in LOCAL_MODEL_CANDIDATES:
            path = Path(candidate)
            if path.is_file():
                return path
        model_name = LOCAL_MODEL_CANDIDATES[-1]

    path = Path(model_name)
    if require_local_model and not path.is_file():
        candidates = ", ".join(LOCAL_MODEL_CANDIDATES)
        raise FileNotFoundError(
            f"YOLO model file '{path}' was not found locally. "
            f"Place one of these files in the working directory or pass an explicit local path: {candidates}. "
            "Refusing to continue because model downloads are disabled."
        )
    return path


def _result_shape(result: Any, frame: Any) -> tuple[int, int]:
    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape is not None and len(orig_shape) >= 2:
        return int(orig_shape[0]), int(orig_shape[1])
    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    raise ValueError("Unable to infer frame shape from YOLO result or frame")


def _box_track_id(box: Any) -> int | None:
    raw_id = getattr(box, "id", None)
    if raw_id is None:
        return None
    try:
        return int(raw_id[0])
    except (TypeError, IndexError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
