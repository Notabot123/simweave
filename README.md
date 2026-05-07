# SimWeave — a hybrid discrete/continuous simulation engine
[![CI](https://github.com/Notabot123/simweave/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Notabot123/simweave/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Notabot123/simweave/branch/main/graph/badge.svg)](https://codecov.io/gh/Notabot123/simweave)
[![PyPI](https://img.shields.io/pypi/v/simweave)](https://pypi.org/project/simweave/)
[![docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://notabot123.github.io/simweave/)

`simweave` is a lightweight Python simulation library built around an
**atomic-clock core** with an **optional heap-based event queue** layered on
top. It is designed for Monte Carlo studies where reproducibility and vectorised
aggregation across replicates matter as much as per-event precision.

**Full documentation:** [notabot123.github.io/simweave](https://notabot123.github.io/simweave/)

Sister project: [EdgeWeave](https://github.com/) — desktop frontend that
generates and runs SimWeave scenarios.

## Design philosophy

### 1. Atomic clock, with an event queue when you need it

The heart of the engine is `simweave.core.environment.SimEnvironment`. It
owns a `Clock` (fixed `dt`), an `EventQueue` (heapq) for sparse future
callbacks, and a list of `Process` objects ticked every step.

```
for t in clock:
    event_queue.fire_due(t, env)       # scheduler-driven work
    for proc in processes:
        proc.tick(dt, env)             # atomic-clock-driven work
    clock.advance(dt)
```

Fixed-step ticking gives Monte Carlo a shared time grid so `N` replicates
aggregate cleanly along a single axis. For scenarios where nothing
happens for long stretches — a factory idle overnight, an agent waiting
for a part — the environment supports `skip_idle_gaps=True` which
fast-forwards to the next scheduled event when every process reports
`has_work(env) == False`. That gives DEVS-like efficiency when demand is
sparse without losing the atomic time grid when it's not.

### 2. Support for Physics and SIUnits

Simweave supports SIunits with dimensional analysis and conversion.
Physical constants are included such as gravity and the speed of light.

```python
from simweave.units.constants import kg, m, s, g, c

force = 10 * kg * m / s**2
weight = 80 * kg * g
energy = 1 * kg * c**2

# show that E=mc²
E = Mass(1.0) * c**2
```

Whilst values are stored in SIUnit they can be instantiated or converted with alternate units.

```python
d = Distance(10, "ft")
print(d.to("m"))       # 3.048
print(d.format("ft"))  # 10.0 [ft]
```

Temperature is supported with notion of absolute and relative (delta).

```python
t1 = Temperature(30, "C")
t2 = Temperature(20, "C")

delta = t1 - t2
```
Finally, we can also use fractional powers when they produce valid physical dimensions
(e.g. √area → distance) using either:
```python
area = sw.Area(np.array([1, 4, 9]))
length = area ** 0.5 # Distance([1.0, 2.0, 3.0]) [m]
length = area.sqrt() # Distance([1.0, 2.0, 3.0]) [m]
```
or indeed:
```python
volume = sw.Volume(np.array([1, 8, 27]))
length = volume ** (1/3) # Distance([1.0, 2.0, 3.0]) [m]
length = volume.cbrt() # Distance([1.0, 2.0, 3.0]) [m]
```

✔ Prevents invalid operations (e.g. distance + time)

✔ Automatically derives units (e.g. m/s → Velocity)

✔ Supports common real-world units (ft, mph, kWh)

### 3. Minimal dependencies

Only `numpy` is a hard dependency. Everything else is optional:

| Extra              | Used for                                                 | Install                              |
|--------------------|----------------------------------------------------------|--------------------------------------|
| `simweave[optim]`  | scipy — DE optimiser, Poisson CDF                        | `pip install simweave[optim]`        |
| `simweave[graph]`  | networkx — interop with external graphs                  | `pip install simweave[graph]`        |
| `simweave[geo]`    | osmnx — OSM map ingestion                                | `pip install simweave[geo]`          |
| `simweave[plot]`   | matplotlib — bring-your-own static plotting              | `pip install simweave[plot]`         |
| `simweave[viz]`    | plotly — first-class viz module (`simweave.viz`)         | `pip install simweave[viz]`          |
| `simweave[ml]`     | pandas — `FaultDataset.to_dataframe()` export            | `pip install simweave[ml]`           |
| `simweave[intl]`   | babel — locale-aware money formatting                    | `pip install simweave[intl]`         |
| `simweave[fast]`   | numba — opt-in JIT for MC inner loops                    | `pip install simweave[fast]`         |
| `simweave[dev]`    | pytest / mypy / plus the optional libs needed for tests  | `pip install simweave[dev]`          |
| `simweave[all]`    | everything above                                         | `pip install simweave[all]`          |

`numpy` stays in core because vectorised inventory / Monte Carlo maths
benefits dramatically from it. Queues still use `collections.deque` for
O(1) push/pop at both ends.

### 4. Modular, but not split into separate libraries

The submodules compose through the shared `Entity` base class:

```
simweave/
  core/          Clock, EventQueue, Entity, SimEnvironment
  units/         SI exponent-tagged units with dimensional analysis
  discrete/      Queue, PriorityQueue, Service, ArrivalGenerator, Resource, ResourcePool
  continuous/    DynamicSystem, RK4/Euler, ContinuousProcess (hybrid)
  faults/        FaultInjector, FaultProfile, FaultDataset — PdM dataset generation
  spatial/       Graph + A*-friendly adj_view adapter, grid_graph helper
  agents/        Compass, routing.a_star, Agent (Entity subclass)
  supplychain/   InventoryItems, Warehouse, steady-state optimisation
  reliability/   ReliableEntity, Fleet, RepairCentre, sensitivity_sweep
  roads/         Road, Intersection, Roundabout, RoadNetwork, traffic signals
  mc/            run_monte_carlo, run_batched_mc
  viz/           Plotly-based plot helpers; applies themes
```

A hybrid sim that moves a pallet (Agent) through a service line (Service
chain) while a mass-spring-damper physics model logs vibration under
load is completely natural in one package; forcing the user to import
three separate libraries to do it would be the wrong trade.

## Installation

### From PyPI (recommended)

```bash
pip install simweave
```

Optional extras (install only what you need):

```bash
pip install simweave[viz]    # plotly charts
pip install simweave[optim]  # scipy optimiser
pip install simweave[ml]     # pandas DataFrame export
pip install simweave[all]    # everything above
```

### From source (latest development)

```bash
git clone https://github.com/Notabot123/simweave
cd simweave
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .[dev]   # editable install with all test deps
pytest                  # should show green
```

## Worked examples

26 runnable scripts live under
[`demos/`](https://github.com/Notabot123/simweave/tree/main/demos):

**Discrete-event / agents / Monte Carlo**

1. `01_simple_queue.py` — M/M/1 queue, Little's law diagnostics
2. `02_chained_services.py` — three-stage production line with blocking
3. `03_resource_pool.py` — bounded concurrency via a physician pool
4. `04_supply_chain_basic.py` — two-echelon SKU inventory with daily demand
5. `05_supply_chain_optimise.py` — Poisson + DE reorder-point optimisation
6. `06_agent_astar.py` — two agents A*-routing on a grid
7. `07_monte_carlo.py` — serial vs threads vs processes vs batched numpy
8. `08_hybrid_continuous_discrete.py` — queue coupled to an MSD

**Continuous dynamical systems**

9. `09_mass_spring_damper.py` — under-, critical-, over-damped responses
10. `10_quarter_car.py` — 2-DOF suspension over a speed bump, three damper setups
11. `11_series_rlc.py` — RLC step and resonant drive, reports ω₀, ζ, Q
12. `12_thermal_system.py` — single-body RC thermal + CPU/heatsink two-mass model
13. `17_half_car_pitch.py` — 4-DOF half-car pitch model
14. `18_half_car_roll.py` — 4-DOF half-car roll model
15. `19_full_car_dynamics.py` — 7-DOF full-car model
16. `20_PID_thermal_system.py` — PID temperature controller

**Currency, visualisation, units, optimisation**

17. `13_money_cashflow.py` — `Money`, FX conversion, locale-aware formatting
18. `14_viz_tour.py` — every `simweave.viz` plot helper end-to-end
19. `15_units_dimensional.py` — SI-units algebra (`Distance / TimeUnit → Velocity`)
20. `16_inventory_optimisation.py` — recover demand rate from a sim, then optimise reorder points
21. `23_time_axis_calendar.py` — calendar-date x-axes on time-series plots

**Reliability and fleet management**

22. `21_reliable_fleet.py` — 8-vehicle taxi fleet with repair centre and spare parts
23. `22_sensitivity_analysis.py` — parameter sweep over repair bays and stock level

**Traffic and road networks**

24. `24_signalised_intersection.py` — four-way signalised junction with traffic signals
25. `25_roundabout.py` — two-lane roundabout with yield logic

**Fault injection and predictive maintenance** *(new in v0.8.0)*

26. `26_fault_injection.py` — inject insulation-loss fault into a thermal model; produce
    a labelled dataset with health index, RUL, and failure-mode columns ready for
    LSTM / time-series model training

Run any script directly from the repo root, e.g.:

```bash
python demos/01_simple_queue.py
python demos/26_fault_injection.py
```

The two-digit prefix is just an ordering hint; scripts have no dependency on each
other. A small `demos/_bootstrap.py` shim adds `src/` to `sys.path` when the
package is not installed; with `pip install -e .` in place the shim is a no-op.

## Monte Carlo performance: picking the right strategy

Monte Carlo scaling on a modern laptop looks like this, ordered by
expected speedup on compute-bound simulations:

| Strategy                       | When to reach for it                                                | Typical speedup vs serial |
|--------------------------------|---------------------------------------------------------------------|----------------------------|
| `run_batched_mc` (numpy)       | You can express the full population as arrays on a shared time grid | 10–100× on vector ops      |
| `executor="processes"`         | Replicates are independent, GIL-bound compute                       | ~N-cores×                  |
| `numba` JIT on hot loops       | Handful of inner functions dominate profile                         | 3–30× on those functions   |
| Cython (or C extension)        | Truly maxing out a known-stable hot path                            | 2–5× over numba            |
| `executor="threads"`           | I/O-bound scenarios (DB hits, REST calls between replicates)        | N-workers×                 |
| Python 3.13 free-threading     | As above, but for compute, once libs stabilise                      | N-cores×                   |

### My recommended ordering

1. **Profile first.** `python -m cProfile demo.py` usually reveals that
   90% of time is in one or two functions. Optimise those, not the
   scheduler.
2. **Vectorise with numpy where possible.** `run_batched_mc` takes a
   `(n_runs, ...)` step function. Any MC where the per-replicate state
   is small and the dynamics are linear-algebraic is a win here. The
   coin-flip toy in `demos/07_monte_carlo.py` runs 10,000 replicates in
   milliseconds this way.
3. **Then reach for `processes`.** `ProcessPoolExecutor` sidesteps the
   GIL without you writing any parallel code. The pickling overhead is
   negligible when each replicate runs for ≥ a second. This is the best
   default for a "complicated simulation" that can't be vectorised.
4. **Then numba.** `@njit` on the inner tick loop often adds 3–30× on
   the hot path. Numba plays nicely with numpy arrays and compiles on
   first call. Prefer it to Cython unless you already have a Cython
   toolchain and need C-level interop — Cython's compile-and-link dance
   is overkill for 95% of cases.
5. **Cython only as a last resort.** If you're building a PyPI wheel
   and every ms counts, Cython can squeeze more out. In practice you
   then have to ship platform-specific wheels, which doubles the CI
   pipeline. Numba's JIT-at-import sidesteps that.
6. **Threads for IO.** The moment your replicate talks to a database,
   an HTTP endpoint, or the filesystem, `executor="threads"` wins
   because while one thread waits on a syscall another runs. On the
   3.13 free-threading build threads will also help compute; treat it
   as a near-term option rather than a today option.

### Seeding and reproducibility

Every Monte Carlo run receives a seed. `run_monte_carlo` materialises
`list(seeds or range(n_runs))` up front and stamps each replicate with
its own `np.random.Generator`, so results are deterministic per seed.
That determinism holds even when replicates are farmed across processes
— every worker sees the same integer seed.

## Packaging, CI, and releases

Continuous integration runs on every push and PR — see `.github/workflows/ci.yml`.
It matrixes `pytest --cov=simweave` across Python 3.10–3.13 on ubuntu / macos /
windows, runs `ruff` and `mypy`, and builds both `sdist` and `wheel` with
`python -m build`. Artefacts are uploaded on every green build and can be
downloaded from the Actions UI for local testing.

`release.yml` publishes on tag push:

- Tags matching a stable semver shape (`v0.1.0`, `v0.2.0`, `v1.0.0`) →
  **PyPI**.
- Tags with a pre-release suffix (`v0.1.0rc1`, `v0.1.0a2`, `v0.1.0-beta`) →
  **TestPyPI**.
- Both use PyPA's trusted publishing (OIDC). No API tokens in GitHub
  secrets — configure once per project on pypi.org / test.pypi.org.

See [`PACKAGING.md`](PACKAGING.md) for the full release checklist and
recommended strategy (pure-Python wheels today; numba for hot paths
before Cython).

## Public API reference

[`SIMWEAVE_API.md`](SIMWEAVE_API.md) is a drop-in reference suitable for
copying into a consuming app's repo (e.g. EdgeWeave). It documents the
top-level imports, the continuous-dynamics protocol for plug-in systems,
and a code-generation template to follow.

## Integration with EdgeWeave

[`EdgeWeave.md`](EdgeWeave.md) describes the flow-based low-code app
that `simweave` is designed to feed. The boundary between the two is
intentionally thin:

- **simweave → EdgeWeave as `@node`s.** Any `simweave` entity with a
  `tick(dt, env)` signature is trivially wrappable as an EdgeWeave node.
  An `ArrivalGenerator` becomes a "Source" node; a `Service` becomes a
  "Process" node; a `Queue` becomes a "Buffer" node. Inputs and outputs
  flow as entity references along drawflow wires.
- **simweave stays headless.** No GUI, no fastapi dependency. EdgeWeave
  imports `simweave` and constructs the environment; `simweave` never
  imports EdgeWeave. This keeps `simweave` useful as a standalone library
  and as a published pypi package.
- **Performance path.** If EdgeWeave wants to JIT hot inner loops,
  enable `simweave[fast]` and let numba do the work — the simulation
  graph stays pure Python while the tight numeric kernels compile
  down. Cython only enters the picture if the EdgeWeave team decides to
  ship prebuilt C extensions for the inner loops, which would be a
  follow-on optimisation rather than a day-one requirement.
- **Serialisation.** Every entity already lives in a dataclass or has
  a short `__init__` signature, so a `to_dict()` / `from_dict()`
  round-trip is a small follow-up when EdgeWeave needs to serialise a
  flow graph.

## Project layout

```
sim_engine/
  src/simweave/          # the package (src layout)
  tests/               # pytest suite (83 tests, scipy-dependent ones skip gracefully)
  demos/               # worked examples (see above)
  archive/             # legacy code kept for reference, safe to delete later
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  EdgeWeave.md         # integration notes for the flow-based front-end
  README.md
```

## Next steps on the roadmap

- **True sim-based optimisation.** `cost_optimise_stock_sim` already
  wraps a user callback; a future step is an adaptive surrogate model
  (kriging / BO) so each evaluation doesn't need a full MC sweep.
- **DEVS-style segment.** When `skip_idle_gaps=True` covers most of the
  gains of a pure event-queue system, a thin DEVS-style adapter on top
  of the event queue would give users who prefer generator-style code
  (à la SimPy) an entry point without forking the kernel.
- **Graph-agnostic spatial layer.** `adj_view` already handles simweave
  Graph, dict-of-dict, and networkx; next is an osmnx adapter so
  `simweave[geo]` lets agents route on real road networks without
  baking osmnx into the core.
- **Numba hot paths.** The A* inner loop and the warehouse vectorisation