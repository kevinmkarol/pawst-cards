# ABOUTME: Watches where a tracked pet has been over a short sliding window so the controller can
# ABOUTME: tell a pet that is walking from one that is settled in place.

import math
from typing import List, Tuple


class Motion:
    def __init__(self, window_seconds: float, threshold_px: float) -> None:
        self.window_seconds = window_seconds
        self.threshold_px = threshold_px
        self._positions: List[Tuple[float, float, float]] = []

    def add(self, x: float, y: float, now: float) -> None:
        """Record the pet's position, in pixels of a frame that does not move with the gimbal."""
        self._positions.append((now, x, y))
        self._forget_old(now)

    def clear(self) -> None:
        self._positions = []

    def moving(self, now: float) -> bool:
        """True when the pet is at least the threshold away from where it was at the window's
        start."""
        self._forget_old(now)
        if len(self._positions) < 2:
            return False
        _, x0, y0 = self._positions[0]
        _, x1, y1 = self._positions[-1]
        return math.hypot(x1 - x0, y1 - y0) >= self.threshold_px

    def _forget_old(self, now: float) -> None:
        self._positions = [p for p in self._positions if now - p[0] <= self.window_seconds]
