# Parameter sweeps and sensitivity analysis

**Status:** proposal for future release (no code yet).

## Motivation

Today, SimWeave can run a single scenario (`simulate()` for continuous
systems, `env.run()` for discrete) and an ensemble of replicates over a
fixed scenario (`run_monte_carlo`, `run_batched_mc`). What it cannot do
out of the box is the *third* axis that practitioners reach for next:
**vary one or more parameters across a grid** and ask how the output
metric changes — the parameter sweep / sensitivity analysis pattern.

Today, users hand-roll this with a nested `for` loop, an ad-hoc dict
of metrics, and a manual matplotlib heatmap. That works but is tedious
and the result shape is bespoke per use case, which defeats the
"fixed-shape outputs aggregate trivially" virtue we lean on for the
Monte Carlo path.

## Goals

1. A **single helper** that takes a scenario function and a parameter
   grid (1-D, 2-D, or N-D) and returns a fixed-shape result array.
2. **Composable with `run_batched_mc`**: each cell of the sweep is
   itself an MC ensemble (so the result shape extends naturally to
   `(n_cells_dim1, n_cells_dim2, ..., n_runs, n_time)` or a
   reduced-statistic version).
3. **Plot helpers in `simweave.viz`** that render the standard
   sensitivity views — heatmaps for 2-D sweeps, tornado charts for
   one-at-a-time perturbation, fan charts for 1-D sweeps with MC.
4. **Backwards compatible.** The hand-rolled nested-loop pattern still
   works; the helper is sugar.
5. **Pure-Python fallback** for the base install. Smarter sampling
   (Sobol sequences, Latin hypercube) lives behind the `[optim]` extra
   alongside scipy.

## Non-goals (initial version)

- Bayesian optimisation / surrogate models. That's a separate
  surrogate-modelling effort and probably belongs in a sister package.
- Adaptive re-sampling on top of an initial grid.
- Distributed sweeps across multiple machines (single-machine
  multiprocessing is enough for now).

## Proposed surface

```python
import simweave as sw

def scenario(params: dict, seed: int) -> dict:
    """Return one or more scalar metrics from a single run."""
    msd = sw.MassSpringDamper(
        mass=params["mass"], damping=params["c"], stiffness=4.0
    )
    res = sw.simulate(msd, t_span=(0.0, 12.0), dt=0.01,
                      x0=[1.0, 0.0])
    return {
        "settling_time": _settling_time(res),
        "peak_displacement": float(np.abs(res.state[:, 0]).max()),
    }

# 2-D sweep, 1 deterministic run per cell.
sweep = sw.parameter_sweep(
    scenario,
    grid={
        "mass":  np.linspace(0.5, 2.0, 16),
        "c":     np.linspace(0.05, 1.0, 12),
    },
    seed=0,
)
# sweep.metrics["settling_time"].shape == (16, 12)

# Same sweep, with 200 MC replicates per cell.
sweep_mc = sw.parameter_sweep(
    scenario,
    grid={"mass": ..., "c": ...},
    n_runs=200,
    seed=0,
    n_workers=8,
)
# sweep_mc.metrics["settling_time"].shape == (16, 12, 200)
```

Plot helpers (in `simweave.viz`):

- `plot_sensitivity_heatmap(sweep, metric, reducer="mean")` — 2-D
  heatmap with axes labelled by the swept parameters.
- `plot_tornado(sweep, metric, baseline)` — bar chart of metric delta
  per parameter perturbation, sorted by impact.
- `plot_sensitivity_fan(sweep_mc, metric, axis)` — collapse all but one
  axis and one metric and draw a percentile fan. Reuses
  `plot_mc_fan` machinery.

## Returned object shape

```python
@dataclass
class SweepResult:
    grid: dict[str, np.ndarray]            # ordered parameter axes
    axes: tuple[str, ...]                  # axis order
    metrics: dict[str, np.ndarray]         # shape == (*grid_shape,) or
                                           # (*grid_shape, n_runs) if MC
    seeds: np.ndarray | None               # only when n_runs > 1
    scenario_name: str
```

This mirrors `SimulationResult` and `MCResult` so the same downstream
JSON serialisation path (and EdgeWeave consumption) works untouched.

## Implementation sketch

- Cartesian product of the grid axes + per-cell seed assignment.
- Single-process loop is the default; `n_workers > 1` dispatches via
  the same `multiprocessing.get_context("spawn").Pool` that
  `run_batched_mc` uses, so we don't introduce a second concurrency
  pattern.
- The scenario callable returns a dict of scalars; missing keys across
  cells become `nan` rather than raising, so partial failures don't
  collapse the whole sweep.
- Reducer functions for the plot helpers operate over the MC axis
  (`mean`, `median`, percentile bands).

## Open questions

- **Sampling strategies.** Should the v0.7 helper take a `sampler=`
  argument (full grid / Sobol / Latin hypercube / one-at-a-time) or
  ship a separate `sw.sample_sobol(...)` that *returns* a parameter
  grid for the helper to consume? Latter feels more composable.
- **Continuous-system convenience.** A common pattern will be sweeping
  ODE-system constructor kwargs. Worth a thin wrapper
  `sw.sweep_system(SystemClass, **swept_kwargs)` that handles the
  scenario-function boilerplate, or is the explicit form clearer?
- **Ranking / reporting.** Should `SweepResult` ship a built-in
  `.rank("metric")` that returns the top-N parameter combinations?
  Tempting but feels like report-builder territory; probably belongs
  in a downstream tool.
- **Memory.** A 100×100 grid of 200-replicate, 1000-step continuous
  simulations is 2 × 10⁹ floats — you cannot keep all the trajectories
  in RAM. The helper should default to *reducing inside the cell*
  (return scalar metrics) and only retain full trajectories on opt-in
  (`return_trajectories=True`) for small sweeps.

## Out of scope but adjacent

- A `sensitivity_indices(...)` helper computing Sobol first-order and
  total-effect indices on top of a sweep result. Useful and
  well-defined but adds a SALib-shaped dependency. Defer.
- Optimisation built on top of sweeps (i.e. "find the parameter
  combination minimising metric X"). The supply-chain module already
  has `cost_optimise_stock_sim` for sim-in-the-loop optimisation; a
  generic `sw.minimise_over_sweep` could subsume it longer term.
