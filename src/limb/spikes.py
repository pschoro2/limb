from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpikeTrace:
    spike_train: np.ndarray
    rates: np.ndarray
    coding: str = "deterministic_rate"

    @property
    def population_activity(self) -> np.ndarray:
        return self.spike_train.mean(axis=0)


class SpikeEncoder:
    """Converts continuous cue vectors into deterministic rate-coded spikes."""

    def __init__(self, time_steps: int = 20) -> None:
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        self.time_steps = time_steps

    def encode(self, vector: np.ndarray) -> SpikeTrace:
        rates = _normalize_to_unit_interval(vector)
        thresholds = (np.arange(self.time_steps, dtype=float) + 0.5) / self.time_steps
        spike_train = (rates[None, :] >= thresholds[:, None]).astype(float)
        return SpikeTrace(spike_train=spike_train, rates=rates)


def _normalize_to_unit_interval(vector: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(vector.astype(float, copy=True), nan=0.0, posinf=1.0, neginf=-1.0)
    values = np.clip(values, -1.0, 1.0)
    return (values + 1.0) / 2.0

