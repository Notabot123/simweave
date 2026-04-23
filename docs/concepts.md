# Concepts

A tour of the four ideas that everything else in SimWeave is built on.

## 1. The `SimEnvironment` is the clock

A `SimEnvironment` owns the simulation clock, the priority event queue,
and a list of registered entities. You pick a fixed tick `dt` and an
end time, then call `env.run()`:

```python
env = sw.SimEnvironment(dt=0.1, end=100.0)
env.register(some_entity)
env.run()                # advances until env.clock.now >= end
env.run(until=50.0)      # or advance only as far as you ask
```

Every tick, the environment:

1. fires any scheduled events whose time is ≤ `now`
2. calls `entity.tick(env)` on every registered entity, in registration
   order
3. advances `now` by `dt`

## 2. Everything ticking is an `Entity`

Queues, services, arrival generators, agents, warehouses and the viz
recorders all subclass `Entity`. To make a new domain object, override
the lifecycle hooks you care about:

```python
class HeartbeatLogger(sw.Entity):
    def __init__(self, name="hb"):
        super().__init__(name=name)
        self.beats = 0

    def on_register(self, env):
        # Called once when the entity joins an environment.
        self.beats = 0

    def tick(self, env):
        # Called every dt while now < end.
        self.beats += 1
```

Registration order matters: an entity registered later sees the
post-tick state of one registered earlier. This is how the viz recorders
work — register them *after* the entity they observe so they snapshot
the post-tick state, not the pre-tick state.

## 3. Continuous systems live alongside discrete ones

`simweave.continuous.simulate` integrates a `DynamicSystem` with a
fixed-step solver (RK4 by default; the `[optim]` extra unlocks SciPy's
adaptive solvers for stiff systems). The result is a `SimulationResult`
with `.time`, `.state`, `.state_labels`, `.system_name`.

You can also wrap a continuous system as a `ContinuousProcess` entity
and register it on a `SimEnvironment`, in which case the integrator
steps once per simulation tick and exposes the latest state via the
process attribute. This is how hybrid sims (continuous physics +
discrete events) are wired up.

## 4. Replication is a first-class citizen

`run_monte_carlo(scenario_fn, n_runs, seed)` (and the parallel
`run_batched_mc`) return an `MCResult` carrying every replicate's
output and the seeds used. Almost every plot helper accepts an
`MCResult`, a raw `(n_runs, n_time)` ndarray, or a `(times, samples)`
tuple — you can bring your own ensemble if you have one already.

## Putting it together

A typical SimWeave program looks like this:

1. Build domain entities (queues, services, agents, warehouses).
2. Build any recorders you want for time-series observation.
3. Construct a `SimEnvironment`, register everything, call `run()`.
4. Read `entity.history` / `recorder.times` etc. or hand them to a plot
   helper for visualisation.
5. Wrap steps 1–4 in a function and pass it to `run_monte_carlo` to get
   a percentile fan around your KPI.
