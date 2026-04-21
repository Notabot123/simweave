"""Monte Carlo over a small queuing sim.

Runs ``run_monte_carlo`` with 32 replicates, each returning the
mean residence time. Demonstrates all three executors: serial,
processes, threads; and also the batched numpy-vectorised style via
``run_batched_mc`` for when you can express the whole population at once.

Run:
    python demos/07_monte_carlo.py
"""
from __future__ import annotations

import time

import numpy as np

from simeng.core.entity import Entity
from simeng.core.environment import SimEnvironment
from simeng.discrete.properties import EntityProperties, exponential
from simeng.discrete.queues import Queue
from simeng.discrete.services import Service, ArrivalGenerator
from simeng.mc.runner import run_monte_carlo, run_batched_mc


def single_queue_mean_wait(seed: int) -> float:
    rng = np.random.default_rng(seed)
    sink = Queue(maxlen=100_000)
    svc = Service(capacity=1, buffer_size=100_000, next_q=sink,
                  default_service_time=1.0, rng=rng)

    def factory(env):
        e = Entity()
        e.sim_properties = EntityProperties(service_time=exponential(1.0))
        return e

    gen = ArrivalGenerator(
        interarrival=lambda r: r.exponential(1.0 / 0.7),
        factory=factory, target=svc, rng=rng,
    )
    env = SimEnvironment(dt=0.25, end=500.0)
    env.register(gen); env.register(svc); env.register(sink)
    env.run()
    return svc.average_wait()


def batched_coin_flips(rng: np.random.Generator, n_runs: int) -> np.ndarray:
    """Toy batched MC: mean of 1000 Bernoulli trials per replicate."""
    return rng.binomial(n=1000, p=0.5, size=n_runs) / 1000.0


def main() -> None:
    n_runs = 16

    for executor in ("serial", "threads", "processes"):
        t0 = time.perf_counter()
        result = run_monte_carlo(
            single_queue_mean_wait, n_runs=n_runs,
            executor=executor, n_workers=4,
            scenario_name=f"mm1-{executor}",
        )
        dt = time.perf_counter() - t0
        arr = np.asarray(result.samples)
        print(f"{executor:<9s}: mean={arr.mean():.3f} std={arr.std():.3f} "
              f"wall={dt:.2f}s")

    print()
    batched = run_batched_mc(batched_coin_flips, n_runs=10_000, seed=0,
                              scenario_name="coin_flips")
    print(f"batched coin flips: mean={batched.samples.mean():.4f} "
          f"std={batched.samples.std():.4f}  (analytical mean=0.5, std=0.0158)")


if __name__ == "__main__":
    main()
