"""Quadcopter motor degradation: when does the controller stop coping?

A FaultInjector gradually weakens the front-right rotor during the figure-8.
The controller has no idea: its integral terms push that rotor harder to
compensate, until it reaches its thrust limit. Beyond that point the vehicle
can no longer hold altitude and attitude together.

Run::

    python demos/29_quadcopter_motor_fault.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

import simweave
from simweave.faults import FaultInjector, FaultProfile, ParameterFault
from simweave.flight import Mission, Quadrotor, fly


def fly_with_fault(loss: float, mission: Mission):
    """Fly with rotor 1 losing ``loss`` of its thrust between t = 12 s and 26 s."""
    quad = Quadrotor()
    if loss == 0.0:
        return fly(quad, mission)
    profile = FaultProfile(onset_time=12.0, failure_time=26.0, mode="rotor_degradation",
                           shape="linear")
    fault = ParameterFault(param="motor_eff_1", profile=profile, max_delta=-loss)
    return fly(FaultInjector(quad, [fault]), mission)


def main():
    print("--- Quadcopter motor degradation ---------------------------")
    mission = Mission.figure8(height=3.0)
    # Rotor 1 must deliver hover thrust / efficiency; the limit is reached at:
    quad = Quadrotor()
    limit = 1.0 - quad.hover_thrust / quad.max_thrust
    print(f"Rotor 1 saturates once it loses about {limit:.0%} of its thrust.\n")

    cases = {"healthy": 0.0, "50% loss": 0.5, "60% loss": 0.6}
    flights = {name: fly_with_fault(loss, mission) for name, loss in cases.items()}

    print(f"{'case':<10s} {'rms err':>8s} {'max err':>8s} {'saturated':>10s} "
          f"{'energy Wh':>10s} {'touchdown':>10s}")
    for name, flight in flights.items():
        m = flight.metrics()
        lowest = flight.position[(flight.time > 8) & (flight.time < 34), 2].min()
        outcome = f"{m['touchdown_speed']:.2f} m/s" if lowest > 0.5 else "crash"
        print(f"{name:<10s} {m['rms_error']:8.3f} {m['max_error']:8.2f} "
              f"{m['saturation_fraction']:10.0%} {m['energy_wh']:10.2f} {outcome:>10s}")

    if not simweave.have_plotly():
        print("\n(Install simweave[viz] to see the plots.)")
        return
    import plotly.graph_objects as go
    from simweave.viz import plot_flight_summary

    fig = go.Figure()
    for name, flight in flights.items():
        fig.add_trace(go.Scatter(x=flight.time, y=flight.position[:, 2], name=name))
    fig.add_vrect(x0=12, x1=26, fillcolor="grey", opacity=0.12, line_width=0,
                  annotation_text="rotor 1 degrading")
    fig.update_layout(title="Altitude under rotor 1 degradation",
                      xaxis_title="time (s)", yaxis_title="altitude (m)")
    fig.show()
    plot_flight_summary(flights["50% loss"], title="50% loss of rotor 1: compensated").show()


if __name__ == "__main__":
    main()
