"""Quadcopter mission: take off, fly a figure-8, land.

A cascaded position / attitude controller flies a 16-state quadrotor model
along a reference mission. Prints tracking metrics per phase and, if plotly is
installed, shows the 3D path.

Run::

    python demos/27_quadcopter_figure8.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

import simweave
from simweave.flight import ConstantWind, Mission, Quadrotor, fly


def main():
    print("--- Quadcopter figure-8 ------------------------------------")

    # 3 m altitude, 8 m x 4 m figure-8 every 20 s, nose along the direction of travel,
    # in a steady 4 m/s wind from the west.
    mission = Mission.figure8(height=3.0, a=4.0, b=2.0, period=20.0, yaw_mode="heading")
    quad = Quadrotor(wind=ConstantWind((4.0, 0.0, 0.0)))

    flight = fly(quad, mission)

    print(f"Mission length: {mission.duration:.0f} s")
    for name, start, end in mission.phases:
        print(f"  {name:<10s} {start:5.1f} -> {end:5.1f} s")

    print("\nMetrics:")
    for key, value in flight.metrics().items():
        print(f"  {key:<20s} {value:8.3f}")

    print("\nRMS tracking error by phase (m):")
    for phase, rms in flight.phase_rms().items():
        print(f"  {phase:<12s} {rms:6.3f}")

    # The columns the EdgeWeave 3D viewer expects (time, x, y, z, roll, pitch, yaw).
    viewer = flight.to_viewer()
    print("\nViewer columns:", ", ".join(viewer))

    if not simweave.have_plotly():
        print("\n(Install simweave[viz] to see the 3D path.)")
        return

    import plotly.graph_objects as go

    pos, ref = flight.position, flight.reference_position
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=ref[:, 0], y=ref[:, 1], z=ref[:, 2], mode="lines",
                               name="reference", line=dict(dash="dash")))
    fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines",
                               name="flown"))
    fig.update_layout(title="Quadcopter figure-8 mission",
                      scene=dict(aspectmode="data",
                                 xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"))
    fig.show()


if __name__ == "__main__":
    main()
