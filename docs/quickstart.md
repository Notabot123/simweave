# Quickstart

Three short, self-contained examples that exercise the main paradigms
SimWeave supports. Every snippet is runnable as-is once you have
`pip install simweave[viz]`.

## 1. Continuous: a damped mass-spring system

```python
import numpy as np
import simweave as sw

msd = sw.MassSpringDamper(m=1.0, c=0.4, k=4.0)
res = sw.simulate(msd, t_span=(0.0, 12.0), dt=0.01, x0=np.array([1.0, 0.0]))

print(res.time.shape, res.state.shape)         # (1201,) (1201, 2)
print(res.state_labels)                        # ('x', 'v')

fig = sw.plot_state_trajectories(res, title="MSD trajectories")
fig.write_html("msd.html", include_plotlyjs="cdn")
```

`simulate()` returns a [`SimulationResult`](modules/continuous.md) with
`.time`, `.state`, `.state_labels` and `.system_name`. Plot helpers
accept any object exposing that interface.

## 2. Discrete: an M/M/c queue

```python
import numpy as np
import simweave as sw

rng = np.random.default_rng(7)
sink = sw.Queue(maxlen=10_000, name="sink")
svc  = sw.Service(
    capacity=2, buffer_size=50, next_q=sink,
    default_service_time=1.0, rng=rng, name="svc",
)
gen = sw.ArrivalGenerator(
    interarrival=lambda r: r.exponential(0.6),
    factory=lambda env: sw.Entity(),
    target=svc, rng=rng, name="gen",
)

# Recorders are optional Entity subclasses; register them after the
# entity they observe so they tick *after* it.
qrec = sw.QueueLengthRecorder(svc, name="svc_qlen")
urec = sw.ServiceUtilisationRecorder(svc, name="svc_util")

env = sw.SimEnvironment(dt=0.1, end=200.0)
for proc in (gen, svc, sink, qrec, urec):
    env.register(proc)
env.run()

sw.plot_queue_length(qrec).write_html("q.html", include_plotlyjs="cdn")
sw.plot_service_utilisation(urec).write_html("u.html", include_plotlyjs="cdn")
```

## 3. Monte Carlo: percentile fan over replications

```python
import numpy as np
import simweave as sw

def scenario(seed):
    rng = np.random.default_rng(seed)
    msd = sw.MassSpringDamper(m=1.0, c=rng.uniform(0.1, 0.6), k=4.0)
    return sw.simulate(msd, t_span=(0.0, 8.0), dt=0.01,
                       x0=np.array([1.0, 0.0]))

mc = sw.run_monte_carlo(scenario, n_runs=200, seed=42)

# Pull the displacement channel from each replicate and fan-chart it.
samples = np.stack([r.state[:, 0] for r in mc.results])
times   = mc.results[0].time

fig = sw.plot_mc_fan((times, samples), title="MSD displacement fan")
fig.write_html("mc.html", include_plotlyjs="cdn")
```

## Where next

- Read [Concepts](concepts.md) to understand the `SimEnvironment` /
  `Entity` model that every example above relies on.
- Browse [Modules](modules/index.md) for module-by-module walkthroughs.
- See the worked examples under `demos/` in the repository for end-to-end
  scripts mirroring each subpackage, including
  [`demos/14_viz_tour.py`](https://github.com/Notabot123/simweave/blob/main/demos/14_viz_tour.py)
  which exercises every plot helper.
