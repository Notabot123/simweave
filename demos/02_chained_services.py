"""A three-stage production line.

    arrivals -> [Inspect] -> [Assemble] -> [Package] -> sink

Each stage has its own capacity and service time. Shows how Service
instances chain via ``next_q`` and how downstream saturation produces
blocking (the middle stage's ``busy_time`` reflects both processing and
blocked-completion time).

Run:
    python demos/02_chained_services.py
"""
from __future__ import annotations

import numpy as np

from simeng.core.entity import Entity
from simeng.core.environment import SimEnvironment
from simeng.discrete.properties import EntityProperties, deterministic, exponential
from simeng.discrete.queues import Queue
from simeng.discrete.services import Service, ArrivalGenerator


def main(horizon: float = 500.0, dt: float = 0.1, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    sink = Queue(maxlen=10_000, name="sink")
    package = Service(capacity=1, buffer_size=5, next_q=sink,
                     default_service_time=1.2, rng=rng, name="package")
    assemble = Service(capacity=2, buffer_size=10, next_q=package,
                      default_service_time=2.0, rng=rng, name="assemble")
    inspect = Service(capacity=1, buffer_size=5, next_q=assemble,
                     default_service_time=0.5, rng=rng, name="inspect")

    def factory(env):
        e = Entity()
        # Each part spends exponential-distributed time at inspect,
        # deterministic elsewhere. This lets the inspect stage show variance.
        e.sim_properties = EntityProperties(service_time=exponential(0.5))
        return e

    gen = ArrivalGenerator(
        interarrival=lambda r: r.exponential(1.0 / 0.8),
        factory=factory, target=inspect, rng=rng, name="gen",
    )

    env = SimEnvironment(dt=dt, end=horizon)
    for proc in (gen, inspect, assemble, package, sink):
        env.register(proc)
    env.run()

    elapsed = env.clock.t - env.clock.start
    print(f"Arrivals                  : {gen.generated}")
    print(f"Items leaving the line    : {len(sink)}")
    for stage in (inspect, assemble, package):
        print(
            f"  {stage.name:>9s}: buffer={len(stage):>2d} "
            f"completed={stage.completed_count:>4d} "
            f"util={stage.utilisation(elapsed):.2f} "
            f"avg_residence={stage.average_residence():.2f}"
        )


if __name__ == "__main__":
    main()
