"""Single-server M/M/1-ish queue.

Arrivals are Poisson (exponential inter-arrivals), service times are
exponential. We run for 2000 time units, then report Little's law
diagnostics alongside the observed utilisation.

Run:
    python demos/01_simple_queue.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

import numpy as np

from simweave.core.entity import Entity
from simweave.core.environment import SimEnvironment
from simweave.discrete.properties import EntityProperties, exponential
from simweave.discrete.queues import Queue
from simweave.discrete.services import Service, ArrivalGenerator


def main(arrival_rate: float = 0.7, service_rate: float = 1.0,
         horizon: float = 2000.0, dt: float = 0.05, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    sink = Queue(maxlen=100_000, name="sink")
    svc = Service(capacity=1, buffer_size=100_000, next_q=sink,
                  default_service_time=1.0 / service_rate, rng=rng, name="svc")

    def factory(env):
        e = Entity()
        e.sim_properties = EntityProperties(
            service_time=exponential(1.0 / service_rate)
        )
        return e

    gen = ArrivalGenerator(
        interarrival=lambda r: r.exponential(1.0 / arrival_rate),
        factory=factory, target=svc, rng=rng, name="gen",
    )

    env = SimEnvironment(dt=dt, end=horizon)
    env.register(gen); env.register(svc); env.register(sink)
    env.run()

    elapsed = env.clock.t - env.clock.start
    L = svc.average_length(elapsed)
    W = svc.average_wait()
    lam_observed = gen.generated / elapsed
    rho = arrival_rate / service_rate

    print(f"Arrivals generated : {gen.generated}")
    print(f"Completions        : {svc.completed_count}")
    print(f"Observed lambda    : {lam_observed:.3f}  (configured {arrival_rate})")
    print(f"Mean queue length L: {L:.3f}")
    print(f"Mean wait      W   : {W:.3f}")
    print(f"Little's L=lam*W   : {lam_observed * W:.3f}")
    print(f"Utilisation        : {svc.utilisation(elapsed):.3f}  (configured rho={rho})")


if __name__ == "__main__":
    main()
