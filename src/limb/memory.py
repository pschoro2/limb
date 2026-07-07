from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Literal, Sequence

import numpy as np

from limb.actions import MotorPrimitive
from limb.spikes import SpikeTrace

MemorySource = Literal["flash", "consolidated"]


@dataclass
class MemoryEntry:
    cue_signature: np.ndarray
    path_trace: SpikeTrace
    action_id: MotorPrimitive
    outcome_valence: float
    risk_score: float
    timestamp: float = field(default_factory=time)
    strength: float = 0.5
    last_used: float = field(default_factory=time)
    source: MemorySource = "flash"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryMatch:
    entry: MemoryEntry
    similarity: float


@dataclass(frozen=True)
class RetrievedMemory:
    matches: tuple[MemoryMatch, ...]

    @property
    def best(self) -> MemoryMatch | None:
        return self.matches[0] if self.matches else None

    @property
    def second(self) -> MemoryMatch | None:
        return self.matches[1] if len(self.matches) > 1 else None

    @property
    def margin(self) -> float:
        if not self.matches:
            return 0.0
        if self.second is None:
            return self.best.similarity if self.best is not None else 0.0
        return max(0.0, self.best.similarity - self.second.similarity)


class MemoryBank:
    """Flash plus consolidated nearest-neighbor memory."""

    def __init__(self, max_flash_entries: int = 256) -> None:
        self.max_flash_entries = max_flash_entries
        self.flash: list[MemoryEntry] = []
        self.consolidated: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        entry.strength = _clamp01(entry.strength)
        if entry.source == "flash":
            self.flash.append(entry)
            if len(self.flash) > self.max_flash_entries:
                self.flash = self.flash[-self.max_flash_entries :]
        elif entry.source == "consolidated":
            self.consolidated.append(entry)
        else:
            raise ValueError(f"Unsupported memory source: {entry.source}")

    def retrieve(self, cue_signature: np.ndarray, top_k: int = 3) -> RetrievedMemory:
        entries = self.flash + self.consolidated
        scored = [
            MemoryMatch(entry=entry, similarity=cosine_similarity(cue_signature, entry.cue_signature))
            for entry in entries
        ]
        scored.sort(key=lambda match: (match.similarity, match.entry.strength), reverse=True)
        now = time()
        for match in scored[:top_k]:
            match.entry.last_used = now
        return RetrievedMemory(matches=tuple(scored[:top_k]))

    def record_episode(
        self,
        cue_signature: np.ndarray,
        path_trace: SpikeTrace,
        action_id: MotorPrimitive,
        outcome_valence: float,
        risk_score: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> MemoryEntry:
        risk = max(0.0, -outcome_valence) if risk_score is None else risk_score
        strength = _initial_strength(outcome_valence)
        entry = MemoryEntry(
            cue_signature=cue_signature.astype(float, copy=True),
            path_trace=path_trace,
            action_id=action_id,
            outcome_valence=float(outcome_valence),
            risk_score=_clamp01(risk),
            strength=strength,
            source="flash",
            metadata=metadata or {},
        )
        self.add(entry)
        return entry

    def consolidate(self, min_strength: float = 0.65) -> list[MemoryEntry]:
        """Promote strong flash memories, with negative outcomes promoted faster."""

        promoted: list[MemoryEntry] = []
        remaining_flash: list[MemoryEntry] = []
        for entry in self.flash:
            threshold = min_strength
            if entry.outcome_valence < 0:
                threshold -= 0.15
            if entry.strength >= threshold:
                clone = MemoryEntry(
                    cue_signature=entry.cue_signature.astype(float, copy=True),
                    path_trace=entry.path_trace,
                    action_id=entry.action_id,
                    outcome_valence=entry.outcome_valence,
                    risk_score=entry.risk_score,
                    timestamp=entry.timestamp,
                    strength=entry.strength,
                    last_used=entry.last_used,
                    source="consolidated",
                    metadata=dict(entry.metadata),
                )
                self.consolidated.append(clone)
                promoted.append(clone)
            else:
                remaining_flash.append(entry)
        self.flash = remaining_flash
        return promoted


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.nan_to_num(left.astype(float, copy=False), nan=0.0, posinf=1.0, neginf=-1.0)
    right = np.nan_to_num(right.astype(float, copy=False), nan=0.0, posinf=1.0, neginf=-1.0)
    if left.shape != right.shape:
        raise ValueError(f"Memory vector shape mismatch: {left.shape} vs {right.shape}")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return _clamp_signed(float(np.dot(left, right) / denom))


def entries_from_sequence(entries: Sequence[MemoryEntry]) -> MemoryBank:
    bank = MemoryBank()
    for entry in entries:
        bank.add(entry)
    return bank


def _initial_strength(outcome_valence: float) -> float:
    if outcome_valence < 0:
        return 0.75
    if outcome_valence > 0:
        return 0.65
    return 0.45


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _clamp_signed(value: float) -> float:
    return min(1.0, max(-1.0, float(value)))
