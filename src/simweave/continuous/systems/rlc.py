from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.units.si import Resistance, Capacitance, Inductance, Current, Voltage

class SeriesRLC(DynamicSystem):
    """Series RLC circuit in charge-current form.

    State x = [q, i]
    Equation:
        L q'' + R q' + (1/C) q = V(t)
    """

    STATE_UNITS = {
        "charge": Voltage,
        "current": Current,      
    }

    def __init__(
        self,
        resistance: float | Resistance,
        inductance: float | Inductance,
        capacitance: float | Capacitance,
        x0: tuple[float, float] = (0.0, 0.0),
    ):
        if self._val(inductance) <= 0 or self._val(capacitance) <= 0:
            raise ValueError("inductance and capacitance must be positive")
        self.resistance = self._val(resistance)
        self.inductance = self._val(inductance)
        self.capacitance = self._val(capacitance)
        self._x0 = np.asarray([self._val(v) for v in x0], dtype=float)

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, str]:
        return ("charge", "current")

    def derivatives(
        self, t: float, state: np.ndarray, inputs: float | int | None = None
    ) -> np.ndarray:
        charge, current = state
        voltage = 0.0 if inputs is None else float(inputs)
        current_dot = (
            voltage - self.resistance * current - charge / self.capacitance
        ) / self.inductance
        return np.array([current, current_dot], dtype=float)
