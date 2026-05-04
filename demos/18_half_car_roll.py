"""Roll-car suspension model demo.

Demonstrates left-right dynamics and roll response to asymmetric road input.

Run::

    python demos/18_half_car_roll.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from simweave.continuous.solver import simulate
from simweave.continuous.systems import RollCarModel

from simweave.viz.vehicle_dynamics import plot_vehicle_metrics

def _show(label: str, value) -> None:
    print(f"{label:<30} {value}")


def main() -> None:
    print("--- Constructing roll-car model ------------------------------")

    model = RollCarModel(
        sprung_mass=1200.0,
        roll_inertia=2200.0,
        unsprung_mass_left=60.0,
        unsprung_mass_right=60.0,
        k_sl=20000.0,
        k_sr=20000.0,
        c_sl=1500.0,
        c_sr=1500.0,
        k_tl=150000.0,
        k_tr=150000.0,
        track_width=1.6,
    )

    print("Model created.")

    print()
    print("--- Road input: left wheel bump ------------------------------")

    def road_input(t: float):
        # only left wheel hits a bump
        z_rl = 0.05 if t > 1.0 else 0.0
        z_rr = 0.0
        return (z_rl, z_rr)

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
    phi = result.state[:, 2]

    _show("Max body displacement", np.max(z_s))
    _show("Max roll angle", np.max(phi))

    print()
    print("Wrap with dimensional units:")
    results_with_units = model.wrap_states(result)
    print("Max sprung displacement:", results_with_units["z_s"].max())
    print("Max sprung velocity:", results_with_units["phi"].max())

    print()
    print("--- Interpretation -------------------------------------------")
    print("• Left bump induces roll to the right")
    print("• Suspension resists roll via stiffness/damping")
    print("• System settles to equilibrium")

    # Optionally pass model to include Tyre Forces else Deflection as default
    fig = plot_vehicle_metrics(
        result,
        model=model
    )

    # Note plot function utilises simweave.analysis.vehicle import compute_vehicle_metrics
    fig.show()


if __name__ == "__main__":
    main()