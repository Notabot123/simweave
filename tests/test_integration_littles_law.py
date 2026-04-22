"""Integration tests that exercise the pipeline end-to-end.

Little's law says that for any stable queuing system in steady state,
    L = lambda * W
where L is the mean number in the system, lambda is the arrival rate, and
W is the mean residence time. We run a long M/M/1-ish simulation and
check the identity holds to a tolerance.
"""

import numpy as np

from simeng.core.entity import Entity
from simeng.core.environment import SimEnvironment
from simeng.discrete.properties import EntityProperties, exponential, deterministic
from simeng.discrete.queues import Queue
from simeng.discrete.services import Service, ArrivalGenerator


def _factory(service_time_dist):
    def make(env):
        e = Entity()
        e.sim_properties = EntityProperties(service_time=service_time_dist)
        return e

    return make


def test_littles_law_single_server_stable():
    # lambda = 0.7, mu = 1.0 -> rho = 0.7, stable.
    rng = np.random.default_rng(123)
    sink = Queue(maxlen=10_000, name="sink")
    svc = Service(
        capacity=1,
        buffer_size=10_000,
        next_q=sink,
        default_service_time=1.0,
        rng=rng,
        name="svc",
    )
    gen = ArrivalGenerator(
        interarrival=lambda r: r.exponential(1.0 / 0.7),
        factory=_factory(exponential(1.0)),
        target=svc,
        rng=rng,
        name="gen",
    )
    env = SimEnvironment(dt=0.05, end=2000.0)
    env.register(gen)
    env.register(svc)
    env.register(sink)
    env.run()

    elapsed = env.clock.t - env.clock.start
    lam_observed = gen.generated / elapsed
    L = svc.average_length(elapsed)
    W = svc.average_wait()
    # Little's law on the buffer portion only.
    assert W > 0
    assert L > 0
    # L == lambda * W up to ~20% due to finite-run noise.
    lhs = L
    rhs = lam_observed * W
    assert abs(lhs - rhs) / max(lhs, rhs) < 0.25


def test_utilisation_tracks_load():
    """Heavier load -> higher measured utilisation."""

    def run(rho):
        rng = np.random.default_rng(7)
        sink = Queue(maxlen=10_000)
        svc = Service(
            capacity=1,
            buffer_size=10_000,
            next_q=sink,
            default_service_time=1.0,
            rng=rng,
        )
        gen = ArrivalGenerator(
            interarrival=lambda r: r.exponential(1.0 / rho),
            factory=_factory(exponential(1.0)),
            target=svc,
            rng=rng,
        )
        env = SimEnvironment(dt=0.1, end=500.0)
        env.register(gen)
        env.register(svc)
        env.register(sink)
        env.run()
        return svc.utilisation(env.clock.t - env.clock.start)

    u_light = run(0.3)
    u_heavy = run(0.8)
    assert u_light < u_heavy
    # Light-load utilisation should be small; heavy close to rho.
    assert u_light < 0.45
    assert u_heavy > 0.6


def test_pipeline_conservation():
    """Items entering == items currently held + items that left the system."""
    sink = Queue(maxlen=10_000, name="sink")
    svc = Service(
        capacity=2,
        buffer_size=10_000,
        next_q=sink,
        default_service_time=1.0,
        name="svc",
    )
    env = SimEnvironment(dt=1.0, end=100.0)
    env.register(svc)
    env.register(sink)

    # Inject a known number of items up front.
    n_inject = 50
    for i in range(n_inject):
        e = Entity(name=f"e{i}")
        e.sim_properties = EntityProperties(service_time=deterministic(1.0))
        svc.enqueue(e)
    env.run()

    in_buffer = len(svc)
    in_channels = sum(1 for ch in svc.channels if ch.is_busy())
    in_sink = len(sink)
    assert in_buffer + in_channels + in_sink == n_inject
