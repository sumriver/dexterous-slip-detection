"""Slip detection via mass estimate fluctuation."""

from __future__ import annotations

from collections import deque

import numpy as np


class SlipDetector:
    """Detect slip when mass estimate deviates from running median.

    Criterion (from research doc):
        |m̃_t - median(m̃)| > threshold
    """

    def __init__(self, window_size: int = 50, threshold: float = 0.15):
        self.window_size = window_size
        self.threshold = threshold
        self._history: deque[float] = deque(maxlen=window_size)

    def update(self, mass_estimate: float) -> bool:
        """Update with new mass estimate; return True if slip detected."""
        if not np.isfinite(mass_estimate):
            return False

        self._history.append(mass_estimate)
        if len(self._history) < 3:
            return False

        median = float(np.median(list(self._history)))
        return abs(mass_estimate - median) > self.threshold

    def reset(self) -> None:
        self._history.clear()

    @property
    def median(self) -> float | None:
        if len(self._history) == 0:
            return None
        return float(np.median(list(self._history)))
