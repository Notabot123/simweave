---
name: simweave
description: "Use this skill whenever the user asks for help building, running, debugging, or visualising simulations with the SimWeave Python library (`pip install simweave`). Triggers include any mention of: simweave, SimWeave, the EdgeWeave companion app, hybrid discrete/continuous simulation, Monte Carlo with `MCResult`/`run_monte_carlo`/`run_batched_mc`, queueing/Service/Queue/PriorityQueue/ArrivalGenerator, Resource/ResourcePool, supply-chain Warehouse/InventoryItems, agent routing on a Graph with A*/dijkstra, continuous dynamic systems (mass-spring-damper, pendulum, RLC, thermal, quarter-car) using `simulate(...)`, `Money` and FX conversion via `simweave.currency`, plotly figures via `simweave.viz`, time-sampled recorders (`QueueLengthRecorder`, `ServiceUtilisationRecorder`, `WarehouseStockRecorder`), or producing plotly JSON for an EdgeWeave frontend. Also use when generating example simulations for documentation or when the user is integrating SimWeave outputs into another tool. Do NOT use this skill for unrelated python work, generic plotting questions, or when the user is working with a different sim library."
---

# SimWeave: hybrid discrete/continuous simulation engine

## What it is

SimWeave (`pip install simweave`) is an atomic-clock simulation engine
that treats discrete events and continuous dynamics as first-class
citizens on the same timeline. Notable subsystems:

| module | purpose |
|---|---|
| `simweave.core` | `Clock`, `EventQueue`, `Entity`, `SimEnvironment` |
| `simweave.discrete` | `Queue`, `PriorityQueue`, `Service`, `ArrivalGenerator`, `Resource`, `ResourcePool`, `EntityProperties`, distributions (`exponential`, `uniform`, `normal`, `deterministic`) |
| `simweave.continuous` | `simulate(...)`, `SimulationResult`, `ContinuousProcess`, ready-made systems (`MassSpringDamper`, `SimplePendulum`, `QuarterCarModel`, `SeriesRLC`, `ThermalRC`, `TwoMassThermal`) |
| `simweave.agents` | `Agent`, `Compass`, `a_star`, `dijkstra`, heuristics (`manhattan`, `euclidean`, `chebyshev`) |
| `simweave.spatial` | `Graph`, `grid_graph` |
| `simweave.supplychain` | `InventoryItems`, `Warehouse` (multi-echelon, `(R, Q)` policy) |
| `simweave.mc` | `MCResult`, `run_monte_carlo`, `run_batched_mc` |
| `simweave.units` | `SIUnit`, `Distance`, `Velocity`, `Acceleration`, `Mass`, `Force`, `Area`, `Volume`, `TimeUnit` |
| `simweave.currency` | `Money`, FX converters, ISO/custom currency registry, `format_money` |
| `simweave.viz` | plotly figure helpers, theme registry, time-sampling recorders |

Every public name is re-exported from the top-level: write
`import simweave as sw` and reach for `sw.Service`, `sw.simulate`,
`sw.plot_state_trajectories`, etc.

## Optional extras

```bash
pip install simweave[viz]    # plotly>=5.18 -- figure helpers
pip install simweave[plot]   # matplotlib  -- legacy/static plots only
pip install simweave[optim]  # scipy        -- supplychain optimisation
pip install simweave[graph]  # networkx     -- richer graph adapters
pip install simweave[geo]    # osmnx        -- street-network graphs
pip install simweave[fast]   # numba        -- JIT speedups
pip install simweave[intl]   # babel        -- locale-aware money formatting
pip install simweave[dev]    # pytest, scipy, networkx, plotly, babel
pip install simweave[all]    # everything except dev tooling
```

The core has only one runtime dependency: `numpy>=1.23`.

## Mental model

* **`SimEnvironment(start, dt, end)`** owns the authoritative clock and
  a process registry. Anything with a `tick(dt, env)` method can be
  registered. Default loop is fixed-step; pass `skip_idle_gaps=True` to
  jump over empty intervals (DEVS-style efficiency).
* **`Entity`** is the base class for every long-lived simulation object.
  Subclasses implement `tick`, optionally `has_work`, optionally
  `on_register`. The age counter advances automatically — always
  `super().tick(dt, env)` first.
* **`SimulationResult`** is the canonical output of `simweave.continuous.simulate`:
  `result.time` (1-D), `result.state` (T x n), `result.state_labels`,
  `result.system_name`, `result.method`.
* **`MCResult`** is the canonical Monte Carlo output:
  `mc.samples` (numpy array or list), `mc.mean()`, `mc.std()`,
  `mc.quantile(q)`, `mc.scenario_name`.

## Idiomatic patterns

### Discrete: an M/M/1-ish queue

```python
import numpy as np
import simweave as sw
from simweave.discrete.properties import EntityProperties, exponential

rng = np.random.default_rng(42)
sink = sw.Queue(maxlen=10_000, name="sink")
svc = sw.Service(capacity=1, buffer_size=10_000, next_q=sink,
                 default_service_time=1.0, rng=rng, name="svc")

def factory(env):
    e = sw.Entity()
    e.sim_properties = EntityProperties(service_time=exponential(1.0))
    return e

gen = sw.ArrivalGenerator(
    interarrival=lambda r: r.exponential(1.4),
    factory=factory, target=svc, rng=rng, name="gen",
)

env = sw.SimEnvironment(dt=0.05, end=2000.0)
env.register_all([gen, svc, sink])
env.run()
print(svc.completed_count, svc.utilisation(env.clock.t))
```

### Continuous: state-space integration

```python
import numpy as np
import simweave as sw

msd = sw.MassSpringDamper(m=1.0, c=0.4, k=4.0)
res = sw.simulate(msd, t_span=(0.0, 12.0), dt=0.01, x0=np.array([1.0, 0.0]))
res.time, res.state, res.state_labels  # ready for plotting
```

### Hybrid: continuous process driven by the SimEnvironment clock

```python
proc = sw.ContinuousProcess(sw.MassSpringDamper(), method="rk4", n_substeps=4)
env = sw.SimEnvironment(dt=0.1, end=10.0)
env.register(proc)
env.run()
res = proc.result()           # SimulationResult, same shape as standalone
```

### Agents on a graph

```python
g = sw.grid_graph(8, 12, diagonal=True)
agent = sw.Agent(graph=g, start_node=(0, 0),
                 tasks=[(7, 11), (3, 4)], speed=2.0,
                 heuristic=sw.manhattan, name="rover")
env = sw.SimEnvironment(dt=0.5, end=50.0)
env.register(agent)
env.run()
agent.history     # [(t, node), ...]
```

### Supply chain

```python
inv = sw.InventoryItems(
    part_names=["widget", "gizmo"],
    unit_cost=[1.0, 2.5],
    stock_level=[20.0, 10.0],
    batchsize=[20.0, 10.0],
    reorder_points=[5.0, 3.0],
    repairable_prc=[0.0, 0.0],
    repair_times=[1.0, 1.0],
    newbuy_leadtimes=[3.0, 5.0],
)
wh = sw.Warehouse(inventory=inv, name="store")
```

### Monte Carlo

Two flavours: scenario-builder (one replicate per call) and batched
(vectorised over all replicates in one numpy op).

```python
import numpy as np
import simweave as sw

# Scenario-builder MC -- portable, per-seed determinism.
def replicate(seed):
    rng = np.random.default_rng(seed)
    return float(rng.normal(0, 1))

mc = sw.run_monte_carlo(replicate, n_runs=500, executor="serial")
mc.mean(), mc.quantile([0.05, 0.5, 0.95])

# Batched MC -- one numpy op for all replicates at once.
def batched(rng, n):
    return rng.normal(0, 1, size=(n, 100))   # (n_runs, n_time)

mc_batched = sw.run_batched_mc(batched, n_runs=10_000, seed=0)
```

For ensemble fan-charts, return `(n_runs, n_time)` so
`sw.plot_mc_fan(mc_batched, times=...)` can render percentile bands.

### Money and FX

```python
from decimal import Decimal
from simweave.currency import Money, StaticFXConverter, format_money

a = Money("100.00", "USD")
b = Money("50.00",  "USD")
(a + b)                         # Money(150.00, USD)
fx = StaticFXConverter({("GBP", "USD"): Decimal("1.27")})
a.to("GBP", fx)                 # Money(78.74, GBP) (banker's rounding)
format_money(a)                 # "$100.00"
```

`Money` is a frozen dataclass; never mutate `.amount` -- build a new
instance. Currency codes are validated against ISO 4217 plus any
codes registered via `register_custom("XBTC", decimals=8)`.

## Visualisation (`simweave.viz`)

All plot helpers return a `plotly.graph_objects.Figure` so JS
frontends (such as **EdgeWeave**) can `fig.to_json()` and render
natively. Themes are applied via a small registry:

```python
import simweave as sw
sw.set_default_theme("dark")             # or "light", "presentation", "minimal"
sw.register_theme("brand", template="plotly_white",
                  palette=("#0a3d62", "#e58e26", "#38ada9"))
sw.set_default_theme("brand")

# Continuous
fig = sw.plot_state_trajectories(res)
fig = sw.plot_phase_portrait(res, x_idx=0, y_idx=1)

# Monte Carlo
fig = sw.plot_mc_fan(mc, times=t_axis,
                     percentiles=(5, 25, 50, 75, 95))
```

### Recorders for discrete/supplychain primitives

`Queue`, `Service` and `Warehouse` track aggregate scalars but **not**
per-tick time series. Wire a recorder to capture history without
touching the core. Register the recorder *after* the entity it
records from — registration order = tick order.

```python
qrec = sw.QueueLengthRecorder(svc)
urec = sw.ServiceUtilisationRecorder(svc)
wrec = sw.WarehouseStockRecorder(wh)
env.register_all([gen, svc, sink, qrec, urec, wrec])
env.run()

sw.plot_queue_length(qrec)
sw.plot_service_utilisation(urec)        # aggregate util + per-channel busy
sw.plot_warehouse_stock(wrec)            # per-SKU lines + reorder dashes
```

### Agents

```python
sw.plot_agent_path(agent, graph=g)       # works for grid graphs (r, c) tuples
                                          # and networkx graphs with .nodes[n]['pos']
```

### EdgeWeave consumption

Every figure is JSON-serialisable. Use `fig.to_json()` for transport,
and on the JS side `Plotly.newPlot(div, data, layout)` (parsed from
the JSON) renders identically. Themes are encoded in the JSON, so
re-themeing on the JS side just requires merging a layout patch.

## Common pitfalls

* **Don't mutate `Money.amount`** — it's frozen. Use arithmetic
  (`a + b`, `a * 2`) or `Money(...)` constructors.
* **Currency arithmetic raises `CurrencyMismatchError`** unless both
  operands share a code. Convert via `.to("GBP", fx)` first.
* **`run_monte_carlo(..., executor="processes")` requires a picklable
  scenario builder** — define it at module level, not inside another
  function.
* **`SimulationResult.state` is shape `(T, n_states)`** — slice
  `state[:, i]` to get a single channel over time.
* **Recorder history needs registration before run** — recorders
  capture nothing if you build them after `env.run()`.
* **Plotly is optional** — `import simweave.viz` is cheap, but calling
  any `plot_*` helper without `simweave[viz]` installed raises a
  clear `ImportError`. Probe with `sw.have_plotly()`.
* **For batched MC fan charts**, ensure your `batched_step` returns
  `(n_runs, n_time)`. The first axis is the replicate axis.
* **`Service` extends `Queue`**, so a Service has a buffer
  (`enqueue` / `len()` / `cumulative_length_time`) plus channels.
  When passing to a recorder, decide whether you want the buffer
  length (`QueueLengthRecorder(svc)`) or the channel utilisation
  (`ServiceUtilisationRecorder(svc)`).

## Reference docs in the repo

* `SIMWEAVE_API.md` — exhaustive API reference, kept in sync with the
  source.
* `CURRENCY_DESIGN.md` — design rationale for the money/FX module.
* `VIZ_DESIGN.md` — design rationale for the viz module.
* `EdgeWeave.md` — companion JS frontend integration notes.
* `PACKAGING.md` — pip install / extras / build notes.
* `demos/14_viz_tour.py` — every plot helper exercised end-to-end.

## When suggesting code

1. Always import the top-level: `import simweave as sw`.
2. For visualisation, prefer the `sw.plot_*` helpers over raw plotly
   calls so themes apply consistently and EdgeWeave gets predictable
   structure.
3. Wire recorders before `env.run()` and place them after the entities
   they sample.
4. Use `MCResult.quantile(...)` for percentiles rather than re-computing
   on `mc.samples` — the helper handles the `axis=0` convention.
5. When unsure of a method, point the user at `SIMWEAVE_API.md` rather
   than guessing — the API is small enough to read in full.
