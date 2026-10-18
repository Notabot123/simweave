"""Quadcopter landing dispersion: Monte Carlo over wind and payload.

Each replicate flies the same take off, figure-8, land mission with a random
steady wind, seeded turbulence and an unknown payload (the controller assumes
the nominal mass). Summarises tracking error, touchdown speed and where the
vehicle lands.

Run::

    python demos/28_quadcopter_wind_montecarlo.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

import simweave
from simweave.flight import Mission, Quadrotor, TurbulentWind, fly
from simweave.mc import run_monte_carlo

MISSION = Mission.figure8(height=3.0, period=20.0)
COLUMNS = ("rms_error", "touchdown_speed", "landing_x", "landing_y", "energy_wh",
           "wind_speed", "payload_factor")


def scenario(seed: int) -> list[float]:
    """One replicate. Module-level so the process pool can pickle it."""
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0.0, 8.0)
    heading = rng.uniform(0.0, 2.0 * np.pi)
    payload = rng.uniform(0.9, 1.3)

    wind = TurbulentWind(
        sigma=1.0, correlation_time=1.5, duration=MISSION.duration + 1.0,
        mean=(speed * np.cos(heading), speed * np.sin(heading), 0.0), rng=rng,
    )
    quad = Quadrotor(wind=wind)
    quad.mass *= payload  # the controller still believes the nominal mass

    flight = fly(quad, MISSION)
    td = flight.touchdown()
    return [
        flight.metrics()["rms_error"], td["vertical_speed"], td["x"], td["y"],
        flight.metrics()["energy_wh"], speed, payload,
    ]


def main():
    n_runs = 24
    print(f"--- Quadcopter Monte Carlo ({n_runs} flights) -------------------")
    mc = run_monte_carlo(scenario, n_runs=n_runs, executor="processes",
                         scenario_name="wind + payload")
    data = np.asarray(mc.samples)
    col = {name: data[:, i] for i, name in enumerate(COLUMNS)}

    miss = np.hypot(col["landing_x"], col["landing_y"])
    print(f"RMS tracking error  median {np.median(col['rms_error']):.3f} m, "
          f"worst {col['rms_error'].max():.3f} m")
    print(f"Touchdown speed     median {np.median(col['touchdown_speed']):.2f} m/s, "
          f"hardest {col['touchdown_speed'].min():.2f} m/s")
    print(f"Landing miss        50% within {np.quantile(miss, 0.5):.2f} m, "
          f"95% within {np.quantile(miss, 0.95):.2f} m")
    print(f"Rotor energy        {col['energy_wh'].mean():.2f} Wh on average "
          f"({col['energy_wh'].min():.2f}-{col['energy_wh'].max():.2f})")
    corr = np.corrcoef(col["payload_factor"], col["energy_wh"])[0, 1]
    print(f"Correlation of payload with energy: {corr:.2f}")

    if not simweave.have_plotly():
        print("\n(Install simweave[viz] to see the landing scatter.)")
        return
    from simweave.viz import plot_landing_dispersion

    plot_landing_dispersion(
        np.column_stack([col["landing_x"], col["landing_y"]]),
        title="Landing dispersion: random wind (0-8 m/s) and payload (+0-30%)",
    ).show()


if __name__ == "__main__":
    main()
