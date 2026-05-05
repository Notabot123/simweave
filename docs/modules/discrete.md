# Discrete event simulation

Queueing-network primitives.

## Building blocks

- `Queue(maxlen, name)` — FIFO buffer of `Entity` instances.
- `PriorityQueue` — min-heap by entity property.
- `Resource` / `ResourcePool` — countable concurrency limiters.
- `Service(capacity, buffer_size, next_q, default_service_time, rng)` —
  multi-channel server that pulls from its own buffer and forwards
  finished entities.
- `ArrivalGenerator(interarrival, factory, target, rng)` — Poisson-by-
  default source that builds entities and pushes them into a target.
- `EntityProperties` — typed attribute bag attached to each entity for
  routing decisions.

## RNG distributions

`exponential`, `uniform`, `normal`, `deterministic` — convenience
factories that return zero-arg samplers bound to a `numpy.random.Generator`.

```python
rng = np.random.default_rng(0)
service_time = sw.exponential(rate=1.0, rng=rng)
service_time()           # callable per draw
```

## Example: M/M/2

See the [Quickstart](../quickstart.md#2-discrete-an-mmc-queue) for the
full snippet. The recorded outputs render as:

<iframe src="../../embeds/queue_length.html"
        width="100%" height="420" frameborder="0"
        loading="lazy"
        title="M/M/2 buffer length"></iframe>

<iframe src="../../embeds/service_util.html"
        width="100%" height="520" frameborder="0"
        loading="lazy"
        title="Service utilisation (2 channels)"></iframe>

## Calendar time axis

Simulation clocks are dimensionless floats.  :class:`~simweave.core.time_axis.SimTimeAxis`
maps any simulation time to a real-world calendar date so that plot axes
show months and years rather than raw tick numbers.

```python
import simweave as sw

# 1 tick = 1 day, simulation starts 1 January 2027
tax = sw.SimTimeAxis(start="2027-01-01", tick_unit="days")

tax.label(90.0)                    # "2027-04-01"
tax.tick_for_date("2027-07-01")    # 181.0  (schedule events by date)

# Pass to any time-series plot helper
fig = sw.plot_queue_length(recorder, time_axis=tax)

# Or apply after the fact
fig = sw.plot_warehouse_stock(recorder)
tax.apply_to_figure(fig)
```

Supported `tick_unit` values: `"seconds"`, `"minutes"`, `"hours"`,
`"days"`, `"weeks"`, `"months"` (≈ 30.44 days), `"years"` (≈ 365.25 days).

Use `tick_size` to express coarser steps — e.g. `tick_unit="hours",
tick_size=4` makes each simulation unit equal to 4 real hours.

::: simweave.core.time_axis.SimTimeAxis

## API

::: simweave.discrete
    options:
      show_root_heading: false
      show_source: true
