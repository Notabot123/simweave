"""Hybrid continuous/discrete simulation.

A queuing system runs alongside a continuous system. The continuous
system is a mass-spring-damper driven by the number of items in the
queue (modelled as an exogenous input). Demonstrates how
``ContinuousProcess`` plugs into the same ``SimEnvironment`` as the
discrete entities.

Run:
    python demos/08_hybrid_continuous_discrete.py
"""
from __future__ import annotations

import numpy as np

from simeng.continuous.solver import ContinuousProcess
from simeng.continuous.systems import MassSpringDamper
from simeng.core.entity import Entity
from simeng.core.environment import SimEnvironment
from simeng.discrete.properties import EntityProperties, exponential
from simeng.discrete.queues import Queue
from simeng.discrete.services import Service, ArrivalGenerator


class QueueDrivenMSD(MassSpringDamper):
    """MSD with forcing proportional to the upstream queue length."""

    def __init__(self, queue: Queue, gain: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.queue = queue
        self.gain = gain

    def derivatives(self, t, state, inputs=None):
        # Apply the queue-length-driven force on top of the base system.
        base = super().derivatives(t, state, inputs=inputs)
        forcing = np.array([0.0, self.gain * len(self.queue)])
        return base + forcing


def main(horizon: float = 100.0, dt: float = 0.1, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    sink = Queue(maxlen=10_000, name="sink")
    svc = Service(capacity=1, buffer_size=50, next_q=sink,
                  default_service_time=1.5, rng=rng, name="svc")

    def factory(env):
        e = Entity()
        e.sim_properties = EntityProperties(service_time=exponential(1.5))
        return e

    gen = ArrivalGenerator(
        interarrival=lambda r: r.exponential(1.2),
        factory=factory, target=svc, rng=rng,
    )

    msd = QueueDrivenMSD(queue=svc, mass=1.0, damping=0.5,
                         stiffness=2.0, x0=(0.0, 0.0))
    proc = ContinuousProcess(msd, n_substeps=4, method="rk4")

    env = SimEnvironment(dt=dt, end=horizon)
    env.register(gen); env.register(svc); env.register(sink); env.register(proc)
    env.run()

    result = proc.result()
    peak_displacement = float(np.abs(result.state[:, 0]).max())
    final_queue_length = len(svc)
    print(f"Simulated {horizon} s. Arrivals={gen.generated}, completions={svc.completed_count}.")
    print(f"Queue length at end: {final_queue_length}")
    print(f"MSD peak displacement: {peak_displacement:.3f}")
    print(f"MSD final state (x, xdot): ({result.state[-1, 0]:.3f}, {result.state[-1, 1]:.3f})")


if __name__ == "__main__":
    main()
