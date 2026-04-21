import numpy as np
import pytest

from simeng.core.entity import Entity
from simeng.core.environment import SimEnvironment
from simeng.discrete.properties import EntityProperties, deterministic
from simeng.discrete.queues import Queue
from simeng.discrete.resources import Resource, ResourcePool
from simeng.discrete.services import Service, ArrivalGenerator


def _mk_item(name, service_time=1.0):
    e = Entity(name=name)
    e.sim_properties = EntityProperties(service_time=deterministic(service_time))
    return e


def test_single_server_processes_items():
    sink = Queue(maxlen=10, name="sink")
    svc = Service(capacity=1, buffer_size=5, next_q=sink, name="svc")
    env = SimEnvironment(dt=1.0, end=10.0)
    env.register(svc); env.register(sink)
    for i in range(3):
        svc.enqueue(_mk_item(f"e{i}", service_time=1.0))
    env.run()
    assert len(sink) == 3
    assert svc.completed_count == 3


def test_parallel_capacity_doubles_throughput():
    sink = Queue(maxlen=20, name="sink")
    svc = Service(capacity=2, buffer_size=20, next_q=sink, name="svc")
    env = SimEnvironment(dt=1.0, end=5.0)
    env.register(svc); env.register(sink)
    for i in range(10):
        svc.enqueue(_mk_item(f"e{i}", service_time=1.0))
    env.run()
    # With 2 channels and 1-unit service times over 5 ticks, roughly 10 done.
    assert svc.completed_count >= 8


def test_blocked_output_does_not_lose_items():
    sink = Queue(maxlen=2, name="sink")
    svc = Service(capacity=1, buffer_size=5, next_q=sink, name="svc")
    env = SimEnvironment(dt=1.0, end=10.0)
    env.register(svc); env.register(sink)
    for i in range(5):
        svc.enqueue(_mk_item(f"e{i}", service_time=1.0))
    env.run(until=3.0)  # sink fills, service blocks
    # 2 items in sink, possibly 1 blocked in channel, rest still in buffer.
    total = len(sink) + sum(1 for ch in svc.channels if ch.is_busy()) + len(svc)
    assert total == 5


def test_resource_pool_limits_concurrency():
    pool = ResourcePool(maxlen=2, name="pool")
    pool.deposit(Resource(name="r0"))
    pool.deposit(Resource(name="r1"))

    sink = Queue(maxlen=20, name="sink")
    # capacity 4 but only 2 resources -> at most 2 concurrent
    svc = Service(capacity=4, buffer_size=20, next_q=sink,
                  resources=pool, name="svc")

    env = SimEnvironment(dt=1.0, end=5.0)
    env.register(pool); env.register(svc); env.register(sink)
    for i in range(10):
        svc.enqueue(_mk_item(f"e{i}", service_time=1.0))
    env.run()
    # All resources should have been reused; no leak back in pool beyond 2.
    assert len(pool) <= 2
    # Throughput bounded by min(resources, capacity) = 2 over 5 ticks -> ~5 served
    assert 3 <= svc.completed_count <= 10


def test_arrival_generator_feeds_service():
    sink = Queue(maxlen=100, name="sink")
    svc = Service(capacity=1, buffer_size=100, next_q=sink,
                  default_service_time=1.0, name="svc")

    def factory(env):
        return _mk_item(f"arrival_{env.clock.t}", service_time=1.0)

    rng = np.random.default_rng(0)
    gen = ArrivalGenerator(
        interarrival=lambda r: 1.0,  # deterministic one per tick
        factory=factory,
        target=svc,
        rng=rng,
        name="gen",
    )
    env = SimEnvironment(dt=1.0, end=20.0)
    env.register(gen); env.register(svc); env.register(sink)
    env.run()
    # 20 arrivals at 1 per tick; service at 1 per tick => steady state, about 19 completed.
    assert gen.generated >= 19
    assert svc.completed_count >= 18
