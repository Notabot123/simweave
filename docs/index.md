# SimWeave

A hybrid discrete/continuous simulation engine for Python. SimWeave gives
you a single environment in which differential equations, queueing systems,
A\* agents over graphs, multi-SKU warehouses, currency-aware money flows
and Monte Carlo replication all share the same clock.

## Why pick SimWeave over a single-paradigm library

Most Python sim libraries do one thing well: SimPy is a process-based
discrete-event scheduler, SciPy gives you ODE integration, NetworkX gives
you graphs, and so on. SimWeave's pitch is the *seam* between them:

- **Hybrid by default.** A continuous mass-spring system, a queueing
  network, and an A\* agent can all share the same `SimEnvironment` and
  the same `dt`. You don't have to bridge two clocks or two event loops.
- **Fixed-shape outputs that aggregate trivially under Monte Carlo.**
  Every `simulate()` call returns a `SimulationResult` with predictable
  array shapes, so 1 000 replicates collapse into a single `(n_runs,
  n_time)` ndarray that the fan-chart helper consumes directly. No
  pandas wrangling, no per-run schema drift.
- **Cheap core, opt-in extras.** The base install is just NumPy. Plotly,
  SciPy, NetworkX, Numba, Babel and OSMnx all sit behind named extras —
  the package stays small even when the surface area is broad.
- **Plotly-first visualisation that round-trips through JSON.** Every
  helper returns a `plotly.graph_objects.Figure`, which means a downstream
  web frontend can render the same chart natively without re-implementing
  it.
- **Composable primitives.** Queues, services, generators, agents,
  warehouses and recorders all subclass one `Entity` interface. New
  domain objects plug in without touching the scheduler.

## Getting started in 60 seconds

```bash
pip install simweave[viz]
```

```python
import numpy as np
import simweave as sw

# Continuous: a damped mass-spring system
msd = sw.MassSpringDamper(mass=1.0, damping=0.4, stiffness=4.0)
res = sw.simulate(msd, t_span=(0.0, 12.0), dt=0.01, x0=np.array([1.0, 0.0]))

# Plot it (requires the [viz] extra)
fig = sw.plot_state_trajectories(res, title="Damped MSD")
fig.write_html("msd.html", include_plotlyjs="cdn")
```

<iframe src="embeds/msd_states.html"
        width="100%" height="480" frameborder="0"
        loading="lazy"
        title="Damped MSD trajectories"></iframe>

## Where to start

- New to the library? Read [Concepts](concepts.md) and then work
  through [Quickstart](quickstart.md).
- Want runnable end-to-end examples? See [Worked demos](#worked-demos)
  below — every subpackage has at least one numbered script in `demos/`.
- Looking for a specific component? The
  [module guide](modules/index.md) walks each subpackage in turn.
- Building a frontend? See
  [EdgeWeave integration](design/edgeweave.md).
- Want the full API? Jump to the [API reference](api.md).

## Worked demos

The [`demos/`](https://github.com/Notabot123/simweave/tree/main/demos)
folder in the repository is the fastest way to see SimWeave end-to-end.
Every script is self-contained — pick one and run it directly, e.g.

```bash
pip install simweave[dev]
python demos/01_simple_queue.py
```

The two-digit prefix is just a stable ordering hint; the scripts have
no dependency on each other.

| Demo                                       | Topic                                   |
| ------------------------------------------ | --------------------------------------- |
| `01_simple_queue.py`                       | M/M/1 queue with arrivals + a sink      |
| `02_chained_services.py`                   | Two services in series                  |
| `03_resource_pool.py`                      | Multi-channel service capacity          |
| `04_supply_chain_basic.py`                 | Warehouse + reorder logic               |
| `05_supply_chain_optimise.py`              | Poisson + DE cost optimisation          |
| `06_agent_astar.py`                        | A\* agent on a grid                     |
| `07_monte_carlo.py`                        | Replication and `MCResult`              |
| `08_hybrid_continuous_discrete.py`         | Continuous physics + discrete events    |
| `09_mass_spring_damper.py`                 | Canonical MSD                           |
| `10_quarter_car.py`                        | Two-mass road model                     |
| `11_series_rlc.py`                         | LCR electrical circuit                  |
| `12_thermal_system.py`                     | Thermal RC network                      |
| `13_money_cashflow.py`                     | `Money` + FX over time                  |
| `14_viz_tour.py`                           | Every plot helper, end-to-end           |
| `15_units_dimensional.py`                  | SI-units algebra (e.g. m/s = velocity)  |
| `16_inventory_optimisation.py`             | Optimisation from estimated demand      |

## Status

SimWeave is currently 0.4.x. The public surface re-exported from
`simweave` is considered stable; submodules may add features but are
unlikely to break existing imports. See the
[design notes](design/index.md) for the rationale behind the major
subsystems.
