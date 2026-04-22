import pytest

from simeng.core.entity import Entity
from simeng.core.environment import SimEnvironment
from simeng.discrete.properties import EntityProperties, exponential
from simeng.discrete.queues import Queue, PriorityQueue


def _item(name, balk=None, renege=None):
    e = Entity(name=name)
    e.sim_properties = EntityProperties(
        service_time=exponential(1.0),
        balk_on_length=balk,
        renege_after=renege,
    )
    return e


def test_enqueue_and_dequeue():
    q = Queue(maxlen=3, name="q")
    a = _item("a")
    b = _item("b")
    c = _item("c")
    assert q.enqueue(a)
    assert q.enqueue(b)
    assert q.enqueue(c)
    assert len(q) == 3
    assert q.is_full
    # Overflow is dropped.
    assert q.enqueue(_item("d")) is False
    assert q.dropped_count == 1
    assert q.dequeue().name == "a"
    assert q.dequeue().name == "b"


def test_forwarding_to_next_queue():
    q2 = Queue(maxlen=2, name="q2")
    q1 = Queue(maxlen=2, next_q=q2, name="q1")
    a = _item("a")
    b = _item("b")
    q1.enqueue(a)
    q1.enqueue(b)
    assert q1.forward() is True
    assert q1.forward() is True
    assert len(q1) == 0
    assert len(q2) == 2


def test_forward_blocks_when_downstream_full():
    q2 = Queue(maxlen=1, name="q2")
    q1 = Queue(maxlen=2, next_q=q2, name="q1")
    q1.enqueue(_item("a"))
    q1.enqueue(_item("b"))
    assert q1.forward() is True  # q2 now has 1
    assert q1.forward() is False  # blocked
    assert len(q1) == 1
    assert len(q2) == 1


def test_balking():
    q = Queue(maxlen=10, name="q")
    for _ in range(3):
        q.enqueue(_item("filler"))
    balker = _item("balker", balk=2)  # max length before balk = 2; len is 3
    assert q.enqueue(balker) is False
    assert q.balked_count == 1


def test_reneging_in_tick_loop():
    q = Queue(maxlen=5, name="q")
    env = SimEnvironment(dt=1.0, end=10.0)
    env.register(q)
    patient = _item("patient")
    impatient = _item("impatient", renege=2.5)
    q.enqueue(patient)
    q.enqueue(impatient)
    env.run(until=5.0)
    # impatient should have reneged after the third tick.
    assert impatient not in list(q)
    assert patient in list(q)
    assert q.reneged_count == 1


def test_queue_metrics_after_run():
    q = Queue(maxlen=5, name="q")
    env = SimEnvironment(dt=1.0, end=10.0)
    env.register(q)
    q.enqueue(_item("a"))
    env.run()
    # Item waited 10 ticks; cumulative_length_time = 10 * dt
    assert q.cumulative_length_time == pytest.approx(10.0)


def test_priority_queue_orders_by_priority():
    pq = PriorityQueue(maxlen=5, name="pq")
    high = _item("high")
    med = _item("med")
    low = _item("low")
    pq.enqueue(low, priority=10)
    pq.enqueue(high, priority=1)
    pq.enqueue(med, priority=5)
    assert pq.dequeue() is high
    assert pq.dequeue() is med
    assert pq.dequeue() is low
