"""Lumped-capacitance thermal models.

Two models live here:

* :class:`ThermalRC` -- a single-body thermal RC circuit. State is a
  single temperature `T` that exchanges heat with an ambient `T_inf`
  through a thermal resistance `R_th`, and stores energy in a thermal
  capacitance `C_th`. Equation:

      C_th * dT/dt = (T_inf - T) / R_th + Q_in(t)

  Useful for batteries, electronics packages, and any first-order
  thermal behaviour.

* :class:`TwoMassThermal` -- two coupled lumps (e.g. a CPU core and a
  heat sink) with internal conductance `k_12` between them and external
  resistance from the second mass to ambient.
"""

from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.continuous.control import PIDController
from simweave.units.si import ThermalCapacitance, ThermalResistance, ThermalConductance, Temperature

class ThermalRC(DynamicSystem):
    """First-order lumped-capacitance thermal model.

    Parameters
    ----------
    thermal_resistance:
        R_th in K/W. Larger => slower equilibration with ambient.
    thermal_capacitance:
        C_th in J/K. Larger => more thermal inertia.
    ambient_temperature:
        T_inf in Kelvin (or any consistent unit). Default 293.15 K (20 C).
    initial_temperature:
        Starting body temperature, same unit as ``ambient_temperature``.
    """

    STATE_UNITS = {
        "temperature": Temperature,
    }

    def __init__(
        self,
        thermal_resistance: float | ThermalResistance,
        thermal_capacitance: float | ThermalCapacitance,
        ambient_temperature: float | Temperature = 293.15,
        initial_temperature: float | Temperature = 293.15,
        controller: None | PIDController = None,
    ) -> None:
        if self._val(thermal_resistance) <= 0 or self._val(thermal_capacitance) <= 0:
            raise ValueError(
                "thermal_resistance and thermal_capacitance must be positive."
            )
        self.R_th = self._val(thermal_resistance)
        self.C_th = self._val(thermal_capacitance)
        self.T_inf = self._val(ambient_temperature)
        self._x0 = np.array([self._val(initial_temperature)], dtype=float)
        self.controller = controller

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, ...]:
        return ("temperature",)

    def derivatives(
        self, t: float, state: np.ndarray, inputs: float | int | None = None
    ) -> np.ndarray:
        T = state[0]
        heat_in = 0.0 if inputs is None else float(inputs)

        if self.controller is not None:
            dt = 1e-3  # or pass properly later from solver
            heat_in += self.controller(T, dt)
        dTdt = ((self.T_inf - T) / self.R_th + heat_in) / self.C_th
        return np.array([dTdt], dtype=float)

    @property
    def time_constant(self) -> float:
        """First-order time constant tau = R_th * C_th (seconds)."""
        return self.R_th * self.C_th


class TwoMassThermal(DynamicSystem):
    """Two-lump thermal model: core <-> sink <-> ambient.

    State: [T_core, T_sink], both in the same unit as ambient.
    Q_in(t) is applied to the core. Sink dissipates to ambient via
    ``R_sink_to_ambient``; core couples to sink via conductance
    ``k_core_to_sink`` (W/K).
    """

    STATE_UNITS = {
        "core_temperature": Temperature,
        "sink_temperature": Temperature,
    }

    def __init__(
        self,
        C_core: float | ThermalCapacitance,
        C_sink: float | ThermalCapacitance,
        k_core_to_sink: float | ThermalConductance,
        R_sink_to_ambient: float | ThermalResistance,
        ambient_temperature: float | Temperature = 293.15,
        initial_core: float | Temperature = 293.15,
        initial_sink: float | Temperature = 293.15,
    ) -> None:
        for name, val in (
            ("C_core", C_core),
            ("C_sink", C_sink),
            ("k_core_to_sink", k_core_to_sink),
            ("R_sink_to_ambient", R_sink_to_ambient),
        ):
            if self._val(val) <= 0:
                raise ValueError(f"{name} must be positive.")
        self.C_core = self._val(C_core)
        self.C_sink = self._val(C_sink)
        self.k_cs = self._val(k_core_to_sink)
        self.R_sa = self._val(R_sink_to_ambient)
        self.T_inf = self._val(ambient_temperature)
        self._x0 = np.array([self._val(initial_core), self._val(initial_sink)], dtype=float)

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, ...]:
        return ("core_temperature", "sink_temperature")

    def derivatives(
        self, t: float, state: np.ndarray, inputs: float | int | None = None
    ) -> np.ndarray:
        T_c, T_s = state
        Q_in = 0.0 if inputs is None else float(inputs)
        q_cs = self.k_cs * (T_c - T_s)  # core -> sink
        q_sa = (T_s - self.T_inf) / self.R_sa  # sink -> ambient
        dTc = (Q_in - q_cs) / self.C_core
        dTs = (q_cs - q_sa) / self.C_sink
        return np.array([dTc, dTs], dtype=float)
