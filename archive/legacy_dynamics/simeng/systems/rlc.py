from __future__ import annotations

import numpy as np

from simeng.solver import DynamicSystem


class SeriesRLC(DynamicSystem):
    """Series RLC circuit in charge-current form.

    State x = [q, i]
    Equation:
        L q¨ + R q˙ + (1/C) q = V(t)
    where i = q˙
    """

    def __init__(self, resistance: float, inductance: float, capacitance: float, x0: tuple[float, float] = (0.0, 0.0)):
        if inductance <= 0 or capacitance <= 0:
            raise ValueError('inductance and capacitance must be positive')
        self.resistance = float(resistance)
        self.inductance = float(inductance)
        self.capacitance = float(capacitance)
        self._x0 = np.asarray(x0, dtype=float)

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, str]:
        return ('charge', 'current')

    def derivatives(self, t: float, state: np.ndarray, inputs: float | int | None = None) -> np.ndarray:
        charge, current = state
        voltage = 0.0 if inputs is None else float(inputs)
        current_dot = (voltage - self.resistance * current - charge / self.capacitance) / self.inductance
        return np.array([current, current_dot], dtype=float)
