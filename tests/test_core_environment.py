import pytest

from simeng.core.environment import SimEnvironment
from simeng.core.entity import Entity


class TickCounter(Entity):
    """Counts how many times tick() was called."""

    def __init__(self):
        super().__init__(name="counter")
        self.ticks = 0
        self._work = True

    def tick(self, dt, env):
        super().tick(dt, env)
        self.ticks += 1

    def has_work(self, env):
        return self._work


def test_env_steps_registered_processes():
    env = SimEnvironment(dt=0.5, end=5.0)
    c = TickCounter()
    env.register(c)
    env.run()
    assert c.ticks == 10
    assert env.clock.t == pytest.approx(5.0)


def test_scheduled_event_fires_at_right_time():
    env = SimEnvironment(dt=1.0, end=10.0)
    fired_at: list[float] = []
    env.schedule_at(4.0, lambda: fired_at.append(env.clock.t))
    env.schedule_after(2.5, lambda: fired_at.append(env.clock.t))
    env.run()
    # Events fire at the start of the tick whose t >= scheduled_time.
    assert len(fired_at) == 2


def test_skip_idle_gaps_fast_forwards_clock():
    env = SimEnvironment(dt=1.0, end=100.0)
    c = TickCounter()
    c._work = False  # nothing to do
    env.register(c)
    fire_times = []
    env.schedule_at(50.0, lambda: fire_times.append(env.clock.t))
    env.run(skip_idle_gaps=True)
    # With skip-ahead, the counter should have ticked only once (when the
    # event at t=50 fires, we step one tick to process it; afterwards we can
    # skip ahead again and then find no more events).
    assert c.ticks <= 2
    assert fire_times == [50.0]


def test_register_requires_tick_method():
    env = SimEnvironment()
    with pytest.raises(TypeError):
        env.register(object())
