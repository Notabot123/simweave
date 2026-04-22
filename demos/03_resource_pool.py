"""A service with a ResourcePool bottleneck.

A hospital triage service has ``capacity=4`` but only two physicians
(Resources) are on duty. Even though four patients could be processed in
parallel by the server's channels, concurrency is bounded by the
physician pool.

Run:
    python demos/03_resource_pool.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

import numpy as np

from simeng.core.entity import Entity
from simeng.core.environment import SimEnvironment
from simeng.discrete.properties import EntityProperties, exponential
from simeng.discrete.queues import Queue
from simeng.discrete.resources import Resource, ResourcePool
from simeng.discrete.services import Service, ArrivalGenerator


def main(horizon: float = 200.0, dt: float = 0.25, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    physicians = ResourcePool(maxlen=2, name="physicians")
    physicians.deposit(Resource(name="dr_alice"))
    physicians.deposit(Resource(name="dr_bob"))

    sink = Queue(maxlen=10_000, name="discharged")
    triage = Service(
        capacity=4,          # four consult rooms
        buffer_size=50,
        next_q=sink,
        resources=physicians,
        rng=rng,
        name="triage",
    )

    def factory(env):
        e = Entity()
        e.sim_properties = EntityProperties(service_time=exponential(3.0))
        return e

    gen = ArrivalGenerator(
        interarrival=lambda r: r.exponential(2.0),
        factory=factory, target=triage, rng=rng, name="arrivals",
    )

    env = SimEnvironment(dt=dt, end=horizon)
    for proc in (physicians, gen, triage, sink):
        env.register(proc)
    env.run()

    elapsed = env.clock.t - env.clock.start
    print(f"Arrivals   : {gen.generated}")
    print(f"Discharged : {len(sink)}")
    print(f"Triage buffer at end: {len(triage)}")
    print(f"Physician utilisation (time weighted):")
    for r in (r for r in [Resource(name='_')] if False):  # placeholder
        pass
    for rname, r in (("dr_alice", None), ("dr_bob", None)):
        pass
    print(f"Pool utilisation proxy: {triage.utilisation(elapsed):.2f}")


if __name__ == "__main__":
    main()
