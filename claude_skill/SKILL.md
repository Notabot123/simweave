---
name: simweave
description: "Use this skill whenever the user asks for help building, running, debugging, or visualising simulations with the SimWeave Python library (`pip install simweave`). Triggers include any mention of: simweave, SimWeave, the EdgeWeave companion app, hybrid discrete/continuous simulation, Monte Carlo with `MCResult`/`run_monte_carlo`/`run_batched_mc`, queueing/Service/Queue/PriorityQueue/ArrivalGenerator, Resource/ResourcePool, supply-chain Warehouse/InventoryItems, agent routing on a Graph with A*/dijkstra, continuous dynamic systems (mass-spring-damper, pendulum, RLC, thermal, quarter-car, half-car, full-car) using `simulate(...)`, PID control via `PIDController`, `Money` and FX conversion via `simweave.currency`, plotly figures via `simweave.viz`, time-sampled recorders (`QueueLengthRecorder`, `ServiceUtilisationRecorder`, `WarehouseStockRecorder`, `RoadOccupancyRecorder`, `IntersectionQueueRecorder`, `FleetAvailabilityRecorder`), calendar date axes via `SimTimeAxis`, fleet operational availability via `simweave.reliability` (ReliableEntity/Fleet/RepairCentre/sensitivity_sweep), road networks via `simweave.roads` (Road/Intersection/Roundabout/TrafficSignal/RoadNetwork), or producing plotly JSON for an EdgeWeave frontend. Also use when generating example simulations for documentation or when the user is integrating SimWeave outputs into another tool. Do NOT use this skill for unrelated python work, generic plotting questions, or when the user is working with a different sim library."
---

# SimWeave: hybrid discrete/continuous simulation engine

## What it is

SimWeave (`pip install simweave`) is an atomic-clock simulation engine
that treats discrete events and continuous dynamics as first-class
citizens on the same timeline. Current version: **0.7.2**.

| Module | Purpose |
|---|---|
| `simweave.core` | `Clock`, `EventQueue`, `Entity`, `SimEnvironment`, `SimTimeAxis` (calendar date axes) |
| `simweave.discrete` | `Queue`, `PriorityQueue`, `Service`, `ArrivalGenerator`, `Resource`, `ResourcePool`, `EntityProperties`, distributions (`exponential`, `uniform`, `normal`, `deterministic`) |
| `simweave.continuous` | `simulate(...)`, `SimulationResult`, `ContinuousProcess`, ready-made systems (`MassSpringDamper`, `SimplePendulum`, `QuarterCarModel`, `HalfCarModel`, `RollCarModel`, `FullCarModel`, `SeriesRLC`, `ThermalRC`, `TwoMassThermal`), PID control (`PIDController`) |
| `simweave.agents` | `Agent`, `Compass`, `a_star`, `dijkstra`, heuristics (`manhattan`, `euclidean`, `chebyshev`) |
| `simweave.spatial` | `Graph`, `grid_graph` |
| `simweave.supplychain` | `InventoryItems`, `Warehouse` (multi-echelon, `(R, Q)` policy) |
| `simweave.reliability` | `SubsystemSpec`, `ReliableEntity`, `Fleet`, `FleetAvailabilityRecorder`, `RepairCentre`, `RepairJob`, `sensitivity_sweep`, `SweepResult` |
| `simweave.roads` | `Road`, `DualCarriageway`, `Vehicle`, `VehicleArrivalProcess`, `TrafficSignal`, `SignalPhase`, `Intersection`, `Roundabout`, `Handedness`, `RoadNetwork`, `RoadOccupancyRecorder`, `IntersectionQueueRecorder` |
| `simweave.mc` | `MCResult`, `run_monte_carlo`, `run_batched_mc` |
| `simweave.units` | `SIUnit`, `Distance`, `Velocity`, `Acceleration`, `Mass`, `Force`, `Area`, `Volume`, `TimeUnit` — supports numpy array operands and physical constants |
| `simweave.currency` | `Money`, FX converters, ISO/custom currency registry, `format_money` |
| `simweave.viz` | plotly figure helpers, theme registry, time-sampling recorders |

Every public name is re-exported from the top-level: write
`import simweave as sw` and reach for `sw.Service`, `sw.simulate`,
`sw.Road`, `sw.Fleet`, `sw.plot_state_trajectories`, etc.

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
  `result.time` (1-D), `result.state` (T × n), `result.state_labels`,
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

msd = sw.MassSpringDamper(mass=1.0, damping=0.4, stiffness=4.0)
res = sw.simulate(msd, t_span=(0.0, 12.0), dt=0.01, x0=np.array([1.0, 0.0]))
res.time, res.state, res.state_labels  # ready for plotting
```

### Continuous: PID control

```python
from simweave.continuous.control.pid import PIDController

pid = PIDController(kp=2.0, ki=0.5, kd=0.1, setpoint=1.0)
# pid.update(measurement, dt) returns the control output each step
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

### Reliability: fleet operational availability

```python
import numpy as np
import simweave as sw
from simweave.reliability import SubsystemSpec, ReliableEntity, Fleet, \
    FleetAvailabilityRecorder, RepairCentre

specs = [
    SubsystemSpec("engine", failure_rate=1/120, sku_index=0,
                  consumable=False, beyond_economic_repair_prc=0.10,
                  repair_time=5.0, unit_cost=8_000.0, repair_cost=2_500.0),
    SubsystemSpec("tyres",  failure_rate=1/45,  sku_index=1,
                  consumable=True, repair_time=0.5, unit_cost=400.0),
]

# ResourcePool must be pre-populated with Resource objects
pool = sw.ResourcePool(maxlen=3, name="technicians")
for i in range(3):
    pool.deposit(sw.Resource(name=f"tech_{i}"))
rc = RepairCentre(capacity=2, resources=pool)

rng = np.random.default_rng(42)
vehicles = [
    ReliableEntity(specs, warehouse=wh, repair_centre=rc,
                   name=f"taxi_{i:02d}",
                   rng=np.random.default_rng(rng.integers(0, 2**32)))
    for i in range(10)
]
fleet = Fleet(vehicles, name="taxi_fleet")
recorder = FleetAvailabilityRecorder(fleet)

env = sw.SimEnvironment(dt=1.0, end=365.0)
env.register(wh)
env.register(rc)
for v in vehicles:
    env.register(v)
env.register(recorder)
env.run()

print(f"Ao = {recorder.mean_operational_availability:.3f}")
print(f"Cost = £{fleet.total_cost:,.0f}")
fig = sw.plot_fleet_availability(recorder, title="Fleet availability")
```

**Sensitivity sweep** — vary one or two parameters, collect a scalar metric:

```python
from simweave.reliability import sensitivity_sweep

def build(n_bays, stock_mult, seed):
    # ... build and run scenario, return scalar metric ...
    return operational_availability

result = sensitivity_sweep(build,
    param1_name="repair_bays", param1_values=[1, 2, 3, 4],
    param2_name="stock_multiplier", param2_values=[0.5, 1.0, 1.5, 2.0],
    metric_name="Ao", n_runs=20)
fig = sw.plot_sensitivity_surface(result, chart_type="heatmap")
```

### Roads: signalised intersection

```python
import numpy as np
import simweave as sw

road_ns = sw.Road(200.0, 13.9, lanes=2, name="NS_approach")
road_ew = sw.Road(200.0, 13.9, lanes=2, name="EW_approach")
exit_ns = sw.Road(200.0, 13.9, name="NS_exit")
exit_ew = sw.Road(200.0, 13.9, name="EW_exit")

signal = sw.TrafficSignal([
    sw.SignalPhase(green_roads=[road_ns], duration=45.0, name="NS_green"),
    sw.SignalPhase(green_roads=[road_ew], duration=30.0, name="EW_green"),
])
junction = sw.Intersection(signal=signal, name="junction")
for road in (road_ns, road_ew):
    junction.add_approach(road)
    road.outlet = junction
junction.add_exit(exit_ns, weight=1.0)
junction.add_exit(exit_ew, weight=1.0)

arrivals = sw.VehicleArrivalProcess(
    interarrival=lambda r: r.exponential(5.0),
    road=road_ns, rng=np.random.default_rng(0),
)
occ_rec = sw.RoadOccupancyRecorder([road_ns, road_ew])
q_rec   = sw.IntersectionQueueRecorder(junction)

net = sw.RoadNetwork()
net.add_signal(signal)          # signals MUST be added before intersections
net.add_intersection(junction)
for r in (road_ns, road_ew, exit_ns, exit_ew):
    net.add_road(r)
net.add_arrival_process(arrivals)
net.add_recorder(occ_rec)
net.add_recorder(q_rec)

env = sw.SimEnvironment(dt=1.0, end=3600.0)
net.register_all(env)           # handles registration order automatically
env.run()

fig = sw.plot_intersection_queues(q_rec, title="Queue lengths")
```

### Roads: roundabout

```python
rb = sw.Roundabout(
    max_circulating=6,
    transit_time=6.0,
    handedness=sw.Handedness.LEFT,   # LEFT=UK/AU/JP, RIGHT=EU/US
    name="roundabout",
)
rb.add_entry(road_north)
rb.add_exit(exit_south, weight=1.0)
road_north.outlet = rb
```

### Calendar date axes with SimTimeAxis

```python
tax = sw.SimTimeAxis(start="2027-01-01", tick_unit="days")
# tick_unit: "seconds","minutes","hours","days","weeks","months","years"
# tick_size: coarser steps, e.g. tick_unit="hours", tick_size=4

tax.label(90.0)                   # "2027-04-01"
tax.tick_for_date("2027-07-01")   # 181.0 — schedule events by named date

# Pass to any time-series plot helper
fig = sw.plot_fleet_availability(recorder, time_axis=tax)
fig = sw.plot_warehouse_stock(wrec, time_axis=tax)
fig = sw.plot_road_occupancy(occ_rec, time_axis=tax)

# Or apply post-hoc to an existing figure
tax.apply_to_figure(fig)
```

### Monte Carlo

```python
import numpy as np
import simweave as sw

# Scenario-builder MC — portable, per-seed determinism.
def replicate(seed):
    rng = np.random.default_rng(seed)
    return float(rng.normal(0, 1))

mc = sw.run_monte_carlo(replicate, n_runs=500, executor="serial")
mc.mean(), mc.quantile([0.05, 0.5, 0.95])

# Batched MC — one numpy op for all replicates at once.
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

`Money` is a frozen dataclass; never mutate `.amount` — build a new
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
fig = sw.plot_vehicle_metrics(res, model=car)

# Monte Carlo
fig = sw.plot_mc_fan(mc, times=t_axis, percentiles=(5, 25, 50, 75, 95))

# Reliability
fig = sw.plot_fleet_availability(recorder, normalize=False, time_axis=tax)
fig = sw.plot_sensitivity_surface(result, chart_type="surface")  # or "heatmap" / "bar"

# Roads
fig = sw.plot_road_occupancy(occ_rec, time_axis=tax)
fig = sw.plot_intersection_queues(q_rec, time_axis=tax)
```

All time-series helpers accept an optional `time_axis=` kwarg taking a
`SimTimeAxis` instance to show calendar dates instead of raw tick numbers.

### Recorders

Register the recorder *after* the entity it records from — registration
order = tick order.

```python
qrec  = sw.QueueLengthRecorder(svc)
urec  = sw.ServiceUtilisationRecorder(svc)
wrec  = sw.WarehouseStockRecorder(wh)
frec  = sw.FleetAvailabilityRecorder(fleet)        # reliability
rrec  = sw.RoadOccupancyRecorder([road_a, road_b]) # roads
iqrec = sw.IntersectionQueueRecorder(junction)     # roads

sw.plot_queue_length(qrec)
sw.plot_service_utilisation(urec)   # aggregate util + per-channel busy
sw.plot_warehouse_stock(wrec)       # per-SKU lines + reorder dashes
sw.plot_fleet_availability(frec)    # stacked area: green/amber/red
sw.plot_road_occupancy(rrec)        # in-transit counts per road
sw.plot_intersection_queues(iqrec)  # total + per-approach queue lengths
```

### EdgeWeave consumption

Every figure is JSON-serialisable. Use `fig.to_json()` for transport,
and on the JS side `Plotly.newPlot(div, data, layout)` (parsed from
the JSON) renders identically. Themes are encoded in the JSON, so
re-theming on the JS side just requires merging a layout patch.

## Common pitfalls

* **`ResourcePool` must be pre-populated** — create with `ResourcePool(maxlen=N)`
  then call `pool.deposit(Resource(name=f"tech_{i}"))` for each slot.
  Do not use `ResourcePool(n=N)` — that argument does not exist.
* **`TrafficSignal` must be registered before its `Intersection`** — the
  signal must update its phase state before the intersection dispatches
  queued vehicles each tick. Use `RoadNetwork.register_all(env)` to
  handle this automatically.
* **`road.outlet` must be set before `env.run()`** — a `None` outlet
  silently drops vehicles from the system.
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
  any `plot_*` helper without `simweave[viz]` installed raises a clear
  `ImportError`. Probe with `sw.have_plotly()`.
* **For batched MC fan charts**, ensure your `batched_step` returns
  `(n_runs, n_time)`. The first axis is the replicate axis.
* **`Service` extends `Queue`** — a Service has a buffer
  (`enqueue` / `len()`) plus channels. Choose `QueueLengthRecorder(svc)`
  for buffer depth or `ServiceUtilisationRecorder(svc)` for channel busy time.
* **Reliability failure rates are per simulation time unit** —
  `failure_rate=1/120` means MTBF of 120 time units. Both time-based
  and cycle-based rates can be active simultaneously on the same subsystem.

## Reference docs in the repo

* `SIMWEAVE_API.md` — exhaustive API reference, kept in sync with the source.
* `CURRENCY_DESIGN.md` — design rationale for the money/FX module.
* `VIZ_DESIGN.md` — design rationale for the viz module.
* `EdgeWeave.md` — companion JS frontend integration notes.
* `PACKAGING.md` — pip install / extras / build notes.
* `demos/14_viz_tour.py` — every plot helper exercised end-to-end.
* `demos/21_reliable_fleet.py` — fleet availability Monte Carlo.
* `demos/22_sensitivity_analysis.py` — 2-D parameter sweep.
* `demos/23_time_axis_calendar.py` — calendar date axes.
* `demos/24_signalised_intersection.py` — signalised 4-way crossroads.
* `demos/25_roundabout.py` — roundabout vs signal comparison.

## When suggesting code

1. Always import the top-level: `import simweave as sw`.
2. For visualisation, prefer the `sw.plot_*` helpers over raw plotly
   calls so themes apply consistently and EdgeWeave gets predictable structure.
3. Wire recorders before `env.run()` and place them after the entities
   they record from in the registration order.
4. For road networks, always use `RoadNetwork.register_all(env)` rather
   than registering components manually — it guarantees correct tick order.
5. For reliability scenarios, pre-populate `ResourcePool` with
   `Resource` objects before passing it to `RepairCentre`.
