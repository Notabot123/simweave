"""Full car suspension model demo.

Demonstrates left-right dynamics and roll response to asymmetric road input.

Run::

    python demos/19_full_car_dynamics.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from simweave.continuous.solver import simulate
from simweave.continuous.systems import FullCarModel


def main():
    print("--- Full car model ------------------------------------------")

    model = FullCarModel(
        1200, 2500, 2200,
        60,
        20000, 1500,
        150000,
        1.2, 1.6,
        1.6
    )

    def road_input(t):
        # asymmetric bump (front-left only)
        return (0.05 if t > 1 else 0.0, 0.0, 0.0, 0.0)

    result = simulate(model, (0.0, 3.0), dt=0.001, inputs=road_input)

    z_s = result.state[:, 0]
    theta = result.state[:, 2]
    phi = result.state[:, 4]

    print("Max heave:", z_s.max())
    print("Max pitch:", theta.max())
    print("Max roll:", phi.max())

if __name__ == "__main__":
    main()