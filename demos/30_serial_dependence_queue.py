"""Bursty arrivals with identical marginals -- serial dependence via a copula.

Scenario
--------
A single-server queue is fed by exponential inter-arrival gaps with mean 1.0
and served in exponential time with mean 0.8 (utilisation 0.8). We run it
twice with the *same marginal* gap distribution:

* **independent** gaps (a Poisson process), and
* **serially dependent** gaps: a lag-1 Gumbel-copula Markov chain, so a short
  gap tends to be followed by another short gap (bursts) and long gaps cluster
  into lulls.

The histograms of the gaps are indistinguishable, but the queue is not.

Run::

    python demos/30_serial_dependence_queue.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import simweave as sw
from simweave.stats import Exponential, MarkovCopulaProcess, kendall_tau

MEAN_GAP = 1.0
MEAN_SERVICE = 0.8
SIM_TIME = 4000.0
DT = 0.05
TAU = 0.6          # lag-1 Kendall tau of the dependent gaps
N_RUNS = 8


def scenario(seed: int, dependent: bool) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    env = sw.SimEnvironment(dt=DT)
    service = sw.Service(capacity=1, buffer_size=100_000, rng=rng)
    if dependent:
        # Build the stateful process per replicate; never share it across runs.
        gaps = MarkovCopulaProcess(sw.GumbelCopula.from_kendall_tau(TAU), Exponential(MEAN_GAP))
    else:
        gaps = sw.exponential(MEAN_GAP)

    times: list[float] = []

    def factory(e):
        times.append(e.clock.t)
        item = sw.Entity()
        item.sim_properties = sw.EntityProperties(service_time=sw.exponential(MEAN_SERVICE))
        return item

    env.register(sw.ArrivalGenerator(gaps, factory, service, rng=rng))
    env.register(service)
    env.run(SIM_TIME)
    observed = np.diff(times)
    lag1 = kendall_tau(observed[:-1], observed[1:])
    return service.average_residence(), float(observed.mean()), lag1


def main() -> None:
    print(f"Utilisation {MEAN_SERVICE / MEAN_GAP:.0%}; {N_RUNS} replicates of {SIM_TIME:.0f} time units\n")
    rows = {}
    for label, dep in (("independent", False), ("serially dependent", True)):
        mc = sw.run_monte_carlo(lambda s, d=dep: scenario(s, d), n_runs=N_RUNS, scenario_name=label)
        arr = np.asarray(mc.samples)
        rows[label] = arr
        se = arr[:, 0].std(ddof=1) / np.sqrt(N_RUNS)
        print(f"{label:20s} mean gap {arr[:, 1].mean():.3f}   lag-1 tau {arr[:, 2].mean():+.2f}   "
              f"time in system {arr[:, 0].mean():5.2f} +/- {se:.2f}")
    ratio = rows["serially dependent"][:, 0].mean() / rows["independent"][:, 0].mean()
    print(f"\nSame marginal gaps, but {ratio:.1f}x the time in system.")

    if not sw.have_plotly():
        print("\n(Install simweave[viz] to see the lag-1 scatter.)")
        return
    import plotly.graph_objects as go

    rng = np.random.default_rng(0)
    dep = MarkovCopulaProcess(sw.GumbelCopula.from_kendall_tau(TAU), Exponential(MEAN_GAP)).sample(rng, 2000)
    fig = go.Figure(go.Scatter(x=dep[:-1], y=dep[1:], mode="markers", marker=dict(size=4, opacity=0.4)))
    fig.update_layout(title="Dependent gaps: each gap against the next",
                      xaxis_title="gap t", yaxis_title="gap t+1")
    fig.show()


if __name__ == "__main__":
    main()
