"""Half-car suspension model demo.

Extends the quarter-car model to include pitch dynamics and
front/rear suspension interaction.

Run::

    python demos/17_half_car.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from simweave.continuous.solver import simulate
from simweave.continuous.systems import HalfCarModel


def _show(label: str, value) -> None:
    print(f"{label:<30} {value}")


def main() -> None:
    print("--- Constructing half-car model ------------------------------")

    model = HalfCarModel(
        sprung_mass=1200.0,
        pitch_inertia=2500.0,
        unsprung_mass_front=60.0,
        unsprung_mass_rear=60.0,
        k_sf=20000.0,
        k_sr=20000.0,
        c_sf=1500.0,
        c_sr=1500.0,
        k_tf=150000.0,
        k_tr=150000.0,
        a=1.2,
        b=1.6,
    )

    print("Model created.")

    print()
    print("--- Road input: front then rear bump --------------------------")

    def road_input(t: float):
        # front hits bump first, rear slightly later
        z_rf = 0.05 if t > 1.0 else 0.0
        z_rr = 0.05 if t > 1.2 else 0.0
        return (z_rf, z_rr)

    print("Simulating...")

    result = simulate(
        model,
        (0.0, 3.0),
        dt=0.001,
        inputs=road_input,
    )

    print()
    print("--- Results summary ------------------------------------------")

    z_s = result.state[:, 0]
    theta = result.state[:, 2]

    _show("Max body displacement", np.max(z_s))
    _show("Max pitch angle", np.max(theta))

    print()
    print("--- Interpretation -------------------------------------------")
    print("• Front bump induces initial pitch")
    print("• Rear bump excites opposite rotation")
    print("• System settles via damping")


if __name__ == "__main__":
    main()