"""Thermal RC with PID control demo.

Demonstrates temperature regulation to a setpoint using a PID controller.

Run::

    python demos/20_thermal_pid_control.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from simweave.continuous.solver import simulate
from simweave.continuous.systems import ThermalRC
from simweave.continuous.control import PIDController
from simweave.viz import plot_state_trajectories
from simweave.units.si import Temperature


def main():
    print("--- Thermal PID control ------------------------------------")

    # target temperature (Kelvin)
    setpoint = 350.0

    pid = PIDController(Kp=8.0, Ki=0.5, Kd=0.0, setpoint=setpoint)

    model = ThermalRC(
        thermal_resistance=0.5,
        thermal_capacitance=100.0,
        ambient_temperature=293.15,
        initial_temperature=293.15,
        controller=pid,
    )

    result = simulate(model, (0.0, 100.0), dt=0.05)

    T = result.state[:, 0]

    print("Final temperature:", T[-1])
    print("Max temperature:", T.max())

    print("\nWrap with dimensional units:")
    wrapped = model.wrap_states(result)
    print("Final temperature:", wrapped["temperature"][-1])

    # plot
    fig = plot_state_trajectories(
        result,
        title="Thermal RC with PID control",
    )
    fig.add_hline(y=setpoint, line_dash="dash", name="setpoint")
    fig.show()

    print("We can also show what happens after a step input disturbance")
    def heat_input(t):
        # step disturbance at t=20
        return 150.0 if t > 20 else 0.0
    
    result = simulate(
        model,
        (0.0, 100.0),
        dt=0.05,
        inputs=heat_input
    )

    # plot
    fig = plot_state_trajectories(
        result,
        title="Thermal RC with PID after disturbance",
    )
    fig.add_vline(x=20, line_dash="dash", annotation_text="disturbance")
    fig.show()


if __name__ == "__main__":
    main()
