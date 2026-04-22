"""Discrete compass (4, 6, 8 or 16 point).

Used by :class:`~simweave.agents.agent.Agent` for cheap orientation tracking
independent of any graph topology.
"""

from __future__ import annotations


class Compass:
    """Discrete compass quantised to a fixed number of points."""

    _LABELS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    _LABELS_4 = ("N", "E", "S", "W")
    _ALLOWED = (4, 6, 8, 16)

    def __init__(self, points: int = 8, angle: float = 0.0) -> None:
        if points not in self._ALLOWED:
            raise ValueError(f"Compass points must be one of {self._ALLOWED}")
        self.points = points
        self.atomic_angle = 360.0 / points
        self.angle = self._quantize(float(angle))

    def _quantize(self, angle: float) -> float:
        angle %= 360.0
        return (round(angle / self.atomic_angle) % self.points) * self.atomic_angle

    @property
    def direction(self) -> str:
        if self.points == 8:
            idx = int(round(self.angle / 45.0)) % 8
            return self._LABELS_8[idx]
        if self.points == 4:
            idx = int(round(self.angle / 90.0)) % 4
            return self._LABELS_4[idx]
        return f"CW_{self.angle:g}"

    def set_absolute(self, angle: float) -> str:
        self.angle = self._quantize(angle)
        return self.direction

    def clockwise(self, delta: float) -> str:
        self.angle = self._quantize(self.angle + delta)
        return self.direction

    def anti_clockwise(self, delta: float) -> str:
        self.angle = self._quantize(self.angle - delta)
        return self.direction
