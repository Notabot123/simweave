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

## API

::: simweave.discrete
    options:
      show_root_hea