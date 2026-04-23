# SimWeave

A hybrid discrete/continuous simulation engine for Python. SimWeave gives
you a single environment in which differential equations, queueing systems,
A\* agents over graphs, multi-SKU warehouses, currency-aware money flows
and Monte Carlo replication all share the same clock.

```python
import numpy as np
import simweave as sw

# Continuous: a damped mass-spring system
msd = sw.MassSpringDamper(m=1.0, c=0.4, k=4.0)
res = sw.simulate(msd, t_span=(0.0, 12.0), dt=0.01, x0=np.array([1.0, 0.0]))

# Plot it (requires the [viz] extra)
fig = sw.plot_state_trajectories(res, title="Damped MSD")
fig.write_html("msd.html", include_plotlyjs="cdn")
```

## Why SimWeave

- **One clock, many paradigms.** Discrete events, continuous ODE
  integration, agents on graphs, supply-chain inventories and currency
  flows tick together in a single `SimEnvironment`.
- **Cheap core.** No heavy dependencies in the base install. NumPy is
  the only required dep; everything else (plotting, optimisation,
  geographic routing, JIT, i18n) sits behind named extras.
- **Designed for downstream tooling.** Every visualisation helper
  returns a `plotly.graph_objects.Figure` that round-trips through
  JSON, so frontends like EdgeWeave can render a simulation result
  natively without re-implementing chart code.
- **Composable primitives.** Queues, services, generators, agents,
  warehouses and recorders all subclass a common `Entity` interface.
  Build new domain objects without touching the scheduler.

## Where to start

- New to the library? Read [Concepts](concepts.md) and then work
  through [Quickstart](quickstart.md).
- Looking for a specific component? The
  [module guide](modules/index.md) walks each subpackage in turn.
- Building a frontend? See
  [EdgeWeave integration](design/edgeweave.md).
- Want the full API? Jump to the [API reference](api.md).

## Status

SimWeave is currently 0.3.x. The public surface re-exported from
`simweave` is considered stable; submodules may add features but are
unlikely to break existing imports. See the
[design notes](design/index.md) for what shipped and what is on the
roadmap.
