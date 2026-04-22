"""Series RLC circuit response to step and sinusoidal voltages.

    L q'' + R q' + (1/C) q = V(t)    with x = [q, i], i = q'

We solve in two scenarios:

1. A 5 V DC step applied at t=0 into an underdamped circuit.
2. A 50 Hz AC source to illustrate resonant response around
   omega_0 = 1/sqrt(L*C).

Reports the natural frequency, damping ratio, quality factor and
the simulated peak current in each case.

Run:
    python demos/11_series_rlc.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from simweave.continuous.solver import simulate
from simweave.continuous.systems import SeriesRLC


def describe_params(R, L, C):
    omega_0 = 1.0 / np.sqrt(L * C)            # rad/s
    f_0 = omega_0 / (2 * np.pi)               # Hz
    zeta = R / 2.0 * np.sqrt(C / L)           # dimensionless
    Q = 1.0 / (R * np.sqrt(C / L))            # quality factor
    print(f"  R={R} ohm, L={L} H, C={C} F")
    print(f"  natural freq      : {f_0:.3f} Hz  (omega_0 = {omega_0:.3f} rad/s)")
    print(f"  damping ratio zeta: {zeta:.3f}")
    print(f"  quality factor Q  : {Q:.3f}")


def main() -> None:
    R, L, C = 1.0, 0.1, 100e-6   # 50 Hz resonant f ~= 50.3 Hz

    print("Step response")
    describe_params(R, L, C)
    step = SeriesRLC(R, L, C, x0=(0.0, 0.0))
    r1 = simulate(step, (0.0, 0.2), dt=5e-5, inputs=lambda t: 5.0)
    i_peak = np.max(np.abs(r1.state[:, 1]))
    i_final = r1.state[-1, 1]
    q_final = r1.state[-1, 0]
    print(f"  peak current     : {i_peak:.3f} A")
    print(f"  current at 200ms : {i_final:.3e} A  (expect -> 0)")
    print(f"  charge at 200ms  : {q_final:.3e} C  (expect -> C * V = {C*5.0:.1e})")
    print()

    print("Sinusoidal drive at resonance (50 Hz)")
    describe_params(R, L, C)
    ac = SeriesRLC(R, L, C, x0=(0.0, 0.0))
    f = 50.0
    r2 = simulate(
        ac, (0.0, 0.5), dt=2e-5,
        inputs=lambda t: 1.0 * np.sin(2 * np.pi * f * t),
    )
    # Take the last 4 periods to estimate steady-state amplitude.
    tail = r2.state[-int(4 * (1 / f) / 2e-5):, 1]
    print(f"  steady-state current amplitude: {np.max(np.abs(tail)):.3f} A")
    print(f"  theoretical at resonance: V / R = {1.0 / R:.3f} A")


if __name__ == "__main__":
    main()
