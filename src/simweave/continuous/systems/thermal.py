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

    def __init__(
        self,
        thermal_resistance: float,
        thermal_capacitance: float,
        ambient_temperature: float = 293.15,
        initial_temperature: float = 293.15,
    ) -> None:
        if thermal_resistance <= 0 or thermal_capacitance <= 0:
            raise ValueError(
                "thermal_resistance and thermal_capacitance must be positive."
            )
        self.R_th = float(thermal_resistance)
        self.C_th = float(thermal_capacitance)
        self.T_inf = float(ambient_temperature)
        self._x0 = np.array([float(initial_temperature)], dtype=float)

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, ...]:
        return ("temperature",)

    def derivatives(
        self, t: float, state: np.ndarray, inputs: float | int | None = None
    ) -> np.ndarray:
        T = state[0]
        heat_in = 0.0 if inputs is None else float(inputs)
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

    def __init__(
        self,
        C_core: float,
        C_sink: float,
        k_core_to_sink: float,
        R_sink_to_ambient: float,
        ambient_temperature: float = 293.15,
        initial_core: float = 293.15,
        initial_sink: float = 293.15,
    ) -> None:
        for name, val in (
            ("C_core", C_core),
            ("C_sink", C_sink),
            ("k_core_to_sink", k_core_to_sink),
            ("R_sink_to_ambient", R_sink_to_ambient),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be positive.")
        self.C_core = float(C_core)
        self.C_sink = float(C_sink)
        self.k_cs = float(k_core_to_sink)
        self.R_sa = float(R_sink_to_ambient)
        self.T_inf = float(ambient_temperature)
        self._x0 = np.array([initial_core, initial_sink], dtype=float)

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
