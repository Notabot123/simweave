"""Mass-spring-damper, three damping regimes.

Classical second-order mechanical system:

    m * x'' + c * x' + k * x = 0

We sweep the damping coefficient across under-, critical-, and
over-damped cases, solve with RK4, and print the salient features of
each response (settling time, overshoot, final value).

Run:
    python demos/09_mass_spring_damper.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from simweave.continuous.solver import simulate
from simweave.continuous.systems import MassSpringDamper


def describe(label: str, m: float, c: float, k: float) -> None:
    zeta = c / (2 * np.sqrt(k * m))
    omega_n = np.sqrt(k / m)
    sys = MassSpringDamper(mass=m, damping=c, stiffness=k, x0=(1.0, 0.0))
    r = simulate(sys, (0.0, 10.0), dt=0.005, method="rk4")
    x = r.state[:, 0]

    # Settling time: time at which |x| stays under 2% of x0 forever after.
    threshold = 0.02
    idx = np.where(np.abs(x) > threshold)[0]
    settle_t = r.time[idx[-1]] if len(idx) else 0.0
    overshoot_pct = max(0.0, (x.min() / x[0] if x[0] != 0 else 0) * -100.0)

    print(f"{label:<14s} m={m}, c={c}, k={k}")
    print(f"  natural freq wn : {omega_n:.3f} rad/s")
    print(f"  damping ratio z : {zeta:.3f}")
    print(f"  settling time   : {settle_t:.2f} s")
    print(f"  max overshoot   : {overshoot_pct:.1f} %")
    print(f"  final x         : {x[-1]: .5f}")
    print()


def main() -> None:
    m = 1.0; k = 4.0           # wn = 2 rad/s
    c_crit = 2 * np.sqrt(k * m)  # = 4
    describe("underdamped",  m=m, c=0.4 * c_crit, k=k)
    describe("critical",     m=m, c=c_crit,        k=k)
    describe("overdamped",   m=m, c=2.0 * c_crit,  k=k)


if __name__ == "__main__":
    main()
