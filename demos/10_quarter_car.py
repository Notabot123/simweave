"""Quarter-car suspension traversing a speed bump.

The two-DOF model captures both sprung (chassis) and unsprung (wheel)
dynamics. We feed it a half-sine "speed bump" road input and report
peak accelerations plus ride comfort metric (RMS of sprung-mass
acceleration over the simulation window).

Run:
    python demos/10_quarter_car.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from simweave.continuous.solver import simulate
from simweave.continuous.systems import QuarterCarModel


def road_input(t: float) -> float:
    """A 10 cm speed bump traversed between t=1 s and t=1.6 s."""
    t0, duration, height = 1.0, 0.6, 0.10
    if t < t0 or t > t0 + duration:
        return 0.0
    phase = np.pi * (t - t0) / duration
    return height * np.sin(phase)


def describe(label: str, sprung=250.0, unsprung=40.0, ks=15_000.0,
             c=1_500.0, kt=200_000.0) -> None:
    model = QuarterCarModel(
        sprung_mass=sprung, unsprung_mass=unsprung,
        suspension_stiffness=ks, damping=c,
        tyre_stiffness=kt,
    )
    r = simulate(model, (0.0, 5.0), dt=0.001, method="rk4", inputs=road_input)
    z_s = r.state[:, 0]
    z_s_dot = r.state[:, 1]

    # Acceleration of sprung mass via finite-difference (for reporting).
    z_s_ddot = np.gradient(z_s_dot, r.time)

    print(f"{label}:")
    print(f"  sprung mass  {sprung:>6.0f} kg, k_s {ks:>7.0f} N/m, c {c:>6.0f} Ns/m")
    print(f"  peak sprung displacement : {np.max(np.abs(z_s)) * 1000:.1f} mm")
    print(f"  peak sprung velocity     : {np.max(np.abs(z_s_dot)):.3f} m/s")
    print(f"  peak sprung acceleration : {np.max(np.abs(z_s_ddot)):.2f} m/s^2")
    print(f"  RMS acceleration (ride)  : {np.sqrt(np.mean(z_s_ddot**2)):.2f} m/s^2")
    print()


def main() -> None:
    describe("soft setup ",   c=800.0)
    describe("stock setup",   c=1_500.0)
    describe("stiff setup",   c=3_500.0)


if __name__ == "__main__":
    main()
