# Monte Carlo

Replication helpers and the `MCResult` container.

## Single-process

```python
import simweave as sw

def scenario(seed):
    rng = np.random.default_rng(seed)
    msd = sw.MassSpringDamper(mass=1.0, damping=rng.uniform(0.1, 0.6), stiffness=4.0)
    return sw.simulate(msd, t_span=(0.0, 8.0), dt=0.01,
                       x0=np.array([1.0, 0.0]))

mc = sw.run_monte_carlo(scenario, n_runs=200, seed=42)
print(mc.n_runs, mc.seeds[:3])
```

## Batched / parallel

```python
mc = sw.run_batched_mc(scenario, n_runs=2_000, seed=42, n_workers=8)
```

`run_batched_mc` shards the work over a process pool. Your scenario
function must be pickleable.

## Plotting an ensemble

The fan chart helper accepts an `MCResult`, a raw 2-D ndarray
`(n_runs, n_time)`, or a `(times, samples)` tuple:

```python
samples = np.stack([r.state[:, 0] for r in mc.results])
times   = mc.results[0].time
sw.plot_mc_fan((times, samples), title="Displacement fan").show()
```

<iframe src="../../embeds/mc_fan.html"
        width="100%" height="500" frameborder="0"
        loading="lazy"
        title="Monte Carlo fan chart"></iframe>

## API

::: simweave.mc
    options:
      show_r