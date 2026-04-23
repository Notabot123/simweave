# Core

The base primitives every other subpackage builds on.

## `SimEnvironment`

Owns the simulation clock, the priority event queue, and the list of
registered entities. Stepped via `env.run()` or `env.run(until=t)`.

```python
import simweave as sw

env = sw.SimEnvironment(dt=0.1, end=10.0)
env.register(my_entity)
env.run()
```

## `Entity`

The base class for anything that lives in a `SimEnvironment`. Override
`on_register(env)` for setup, `tick(env)` for per-step work, and
`on_unregister(env)` for teardown. Entities also carry a `history`
list that subclasses populate however they like.

## `Clock` and `EventQueue`

Lower-level building blocks. Most users never touch these directly —
the `SimEnvironment` wraps them. They become useful when you want to
schedule a one-shot event in the future:

```python
env.schedule(at=env.clock.now + 5.0, event=ScheduledEvent(callback=fn))
```

## API

::: simweave.core
    options:
      show_root_heading: false
      show_source: true
