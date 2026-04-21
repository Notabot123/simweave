"""Thermal lumped-capacitance models.

Two scenarios that should be familiar to engineering students:

1. A 100 W load switched on in an electronics package. The single-body
   ``ThermalRC`` model gives a classic exponential rise toward
   T_inf + P * R_th.

2. A CPU under a 60 s square-wave burst load. Two-body ``TwoMassThermal``
   shows the core spiking fast while the heatsink warms slowly, which
   is why thermal designers size for sustained load on the sink but
   for peak transient on the core.

Run:
    python demos/12_thermal_system.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from simeng.continuous.solver import simulate
from simeng.continuous.systems import ThermalRC, TwoMassThermal


def scenario_1_single_body() -> None:
    print("=== Single-body ThermalRC ===")
    # Electronics package: R = 2 K/W, C = 100 J/K -> tau = 200 s
    sys = ThermalRC(
        thermal_resistance=2.0,
        thermal_capacitance=100.0,
        ambient_temperature=293.15,
        initial_temperature=293.15,
    )
    print(f"time constant tau = R*C = {sys.time_constant:.1f} s")

    # Step in heat: 100 W continuous
    power = 100.0
    r = simulate(sys, (0.0, 1500.0), dt=1.0, inputs=lambda t: power)
    T = r.state[:, 0] - 273.15  # convert to Celsius for readability

    T_ss_expected = (293.15 + power * sys.R_th) - 273.15
    print(f"temperature at t=tau : {T[int(sys.time_constant)]:.1f} C  "
          f"(expect ~{T_ss_expected * (1 - np.exp(-1)):.1f} C above ambient)")
    print(f"temperature at t=5*tau: {T[-1]:.1f} C  (expect {T_ss_expected:.1f} C)")
    print()


def scenario_2_cpu_sink() -> None:
    print("=== Two-mass: CPU core + heatsink ===")
    # Core: small capacitance, tightly coupled to sink
    # Sink: bigger capacitance, loose coupling to ambient
    sys = TwoMassThermal(
        C_core=5.0,          # J/K -- small
        C_sink=400.0,        # J/K -- bigger
        k_core_to_sink=2.0,  # W/K
        R_sink_to_ambient=0.3,  # K/W
        ambient_temperature=293.15,
        initial_core=293.15,
        initial_sink=293.15,
    )

    def load(t: float) -> float:
        """Square-wave: 80 W for 60 s on, 60 s off, repeating."""
        phase = (t % 120.0) / 120.0
        return 80.0 if phase < 0.5 else 5.0  # idle draws 5 W

    r = simulate(sys, (0.0, 600.0), dt=0.5, inputs=load)
    T_core = r.state[:, 0] - 273.15
    T_sink = r.state[:, 1] - 273.15

    print(f"Peak core T      : {T_core.max():.1f} C")
    print(f"Peak sink T      : {T_sink.max():.1f} C")
    print(f"Core - Sink delta: {np.max(T_core - T_sink):.1f} C  (hottest moment)")
    print(f"Final core T     : {T_core[-1]:.1f} C")
    print(f"Final sink T     : {T_sink[-1]:.1f} C")


def main() -> None:
    scenario_1_single_body()
    scenario_2_cpu_sink()


if __name__ == "__main__":
    main()
