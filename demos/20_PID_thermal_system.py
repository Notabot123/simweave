"""Thermal RC with PID control demo.

Demonstrates temperature regulation to a setpoint with a PID controller run as a
sampled-data controller: ``simulate(..., controller=pid, control_dt=...)`` calls
the PID once per control period and holds its output (heater power, in watts)
until the next sample.

(``ThermalRC(controller=...)`` still works but is deprecated; this is the
supported way.)

Run::

    python demos/20_PID_thermal_system.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from simweave.continuous.solver import simulate
from simweave.continuous.systems import ThermalRC
from simweave.continuous.control import PIDController
from simweave.viz import plot_state_trajectories


def make_plant() -> ThermalRC:
    return ThermalRC(
        thermal_resistance=0.5,
        thermal_capacitance=100.0,
        ambient_temperature=293.15,
        initial_temperature=293.15,
    )


class WithDisturbance:
    """Adds an unmodelled heat load to the controller's output."""

    def __init__(self, controller, load_watts, start):
        self.controller, self.load, self.start = controller, load_watts, start

    def reset(self):
        self.controller.reset()

    def update(self, t, state, dt):
        return self.controller.update(t, state, dt) + (self.load if t > self.start else 0.0)


def main():
    print("--- Thermal PID control ------------------------------------")

    # target temperature (Kelvin)
    setpoint = 350.0
    pid = PIDController(Kp=8.0, Ki=0.5, setpoint=setpoint)

    model = make_plant()
    result = simulate(model, (0.0, 100.0), dt=0.05, controller=pid, control_dt=0.5)

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
    disturbed = WithDisturbance(PIDController(Kp=8.0, Ki=0.5, setpoint=setpoint), 150.0, 20.0)
    result = simulate(make_plant(), (0.0, 100.0), dt=0.05, controller=disturbed, control_dt=0.5)

    # plot
    fig = plot_state_trajectories(
        result,
        title="Thermal RC with PID after disturbance",
    )
    fig.add_vline(x=20, line_dash="dash", annotation_text="disturbance")
    fig.show()


if __name__ == "__main__":
    main()
