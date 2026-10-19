"""Same marginals, same Kendall tau -- does the copula family matter?

Scenario
--------
Jobs arrive with mean gap 1.0 and need exponential service with mean 0.8.
The job's service time and the gap that *follows* it are negatively dependent
(a big job tends to be followed by a short gap, so work piles up). We compare:

* independence,
* Gaussian, Clayton and Gumbel copulas, each with Kendall tau = -0.5 (Clayton
  and Gumbel with the gap axis flipped so the dependence is negative).

All four share identical marginals. Dependence itself clearly raises waits; the
family (tail shape) matters much less at this load, which is why we report
confidence intervals rather than single runs. Student-t needs scipy
(``pip install simweave[optim]``) and is included when it is installed.

Run::

    python demos/31_copula_families_queue.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import simweave as sw
from simweave.stats import Exponential, JointSampler

MEAN_GAP = 1.0
MEAN_SERVICE = 0.8
SIM_TIME = 4000.0
DT = 0.05
TAU = 0.5
N_RUNS = 12


def copulas() -> dict[str, object]:
    out = {
        "independent": sw.GaussianCopula(1e-9),
        "gaussian": sw.GaussianCopula.from_kendall_tau(-TAU),
        "clayton (flipped)": sw.ClaytonCopula.from_kendall_tau(TAU).flipped((0,)),
        "gumbel (flipped)": sw.GumbelCopula.from_kendall_tau(TAU).flipped((0,)),
    }
    try:
        out["student-t"] = sw.StudentTCopula.from_kendall_tau(-TAU, df=4.0)
    except ImportError:
        pass
    return out


def scenario(seed: int, copula) -> float:
    rng = np.random.default_rng(seed)
    env = sw.SimEnvironment(dt=DT)
    service = sw.Service(capacity=1, buffer_size=100_000, rng=rng)
    joint = JointSampler(
        copula, {"interarrival": Exponential(MEAN_GAP), "service_time": Exponential(MEAN_SERVICE)}
    )

    def factory(e, draw):
        item = sw.Entity()
        item.sim_properties = sw.EntityProperties.from_draw(draw)
        return item

    # "after": the service time is paired with the gap that follows the job.
    env.register(sw.ArrivalGenerator(None, factory, service, rng=rng, joint=joint, gap_alignment="after"))
    env.register(service)
    env.run(SIM_TIME)
    return service.average_residence()


def main() -> None:
    print(f"{N_RUNS} replicates of {SIM_TIME:.0f} time units, utilisation {MEAN_SERVICE / MEAN_GAP:.0%}\n")
    results = {}
    for name, cop in copulas().items():
        mc = sw.run_monte_carlo(lambda s, c=cop: scenario(s, c), n_runs=N_RUNS, scenario_name=name)
        x = np.asarray(mc.samples, dtype=float)
        results[name] = x
        print(f"{name:20s} time in system {x.mean():5.2f} +/- {1.96 * x.std(ddof=1) / np.sqrt(N_RUNS):.2f} (95% CI)")

    if not sw.have_plotly():
        print("\n(Install simweave[viz] to see the comparison.)")
        return
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, x in results.items():
        fig.add_trace(go.Box(y=x, name=name))
    fig.update_layout(title="Time in system by dependence structure (same marginals)",
                      yaxis_title="mean time in system")
    fig.show()


if __name__ == "__main__":
    main()
