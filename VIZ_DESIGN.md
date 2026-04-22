# Design note: `simweave.viz`

Status: **proposal**.
Purpose: give users one-liner plotting helpers that turn the standard
simulation outputs — `SimulationResult`, `MCResult`, `Queue`/`Service`
histories, agent `Agent.history`, `Warehouse` stock trajectories — into
the plots a simulation engineer reaches for most often.

---

## Extras vs separate library — my recommendation

**Ship as a submodule under a `simweave[viz]` extra first.** Split into
`simweave-viz` only if specific triggers fire later.

### Why extras first

1. **Cohesion.** Every plot helper touches simweave's internal data
   models (`SimulationResult.state`, `Service.utilisation(…)`, etc.).
   Splitting them means duplicating those types in a "protocol" layer
   or pulling simweave as a dep of the viz package — either way you pay
   for the split.
2. **Discovery.** `pip install simweave[viz]` is one line and users
   already know simweave. A separate `simweave-viz` on PyPI adds a
   discovery step and a version-pinning puzzle (does `simweave-viz 0.2`
   work with `simweave 0.1`?).
3. **matplotlib is already optional.** `simweave[plot]` exists already.
   Nothing forces headless users to install it. So the core stays
   slim regardless.
4. **Low initial surface.** Five or six plot functions will not
   justify an independent release cadence, an independent CI, or its
   own docs site.

### Triggers that would justify splitting it out later

- You want **multiple viz backends** (matplotlib, plotly, bokeh,
  altair) as independent packages. An extras matrix with five
  backends gets ugly; separate backend packages (`simweave-viz-plotly`,
  `simweave-viz-mpl`) keep install graphs clean.
- A dedicated **viz team** wants to iterate on their own release
  schedule.
- The viz code grows a **genuine dependency footprint** (pandas,
  bokeh server components, xarray, panel) that core simweave users
  shouldn't incur.
- You want to produce **interactive dashboards** (panel / streamlit /
  dash) which pull in a web runtime. That absolutely belongs in a
  separate package.

None of those triggers apply yet. Start small.

---

## Proposed surface (`simweave.viz`)

Convention: every function **returns the `matplotlib.axes.Axes`** (or a
list of axes) so users can further customise. Each also accepts an
optional `ax=` argument to plug into existing figures.

```python
from simweave.viz import (
    plot_state_trajectories,
    plot_phase_portrait,
    plot_queue_length,
    plot_service_utilisation,
    plot_mc_fan,
    plot_warehouse_stock,
    plot_agent_path,
)
```

### `plot_state_trajectories(result, channels=None, ax=None)`

Line plot of each state channel vs. `result.time`, labelled with
`result.state_labels`. One-liner on top of `SimulationResult`.

```python
from simweave.viz import plot_state_trajectories
plot_state_trajectories(res)               # all channels on shared axes
plot_state_trajectories(res, channels=(0,)) # just the first
```

### `plot_phase_portrait(result, x_idx=0, y_idx=1, ax=None)`

For 2-state systems (MSD, pendulum, RLC). Plots state[x] vs state[y]
with an arrow showing time direction and a marker at the start state.

### `plot_queue_length(queue, env, ax=None)`

Reads a `Queue`'s time-sampled length history and plots it. Requires
the queue was registered with a `SimEnvironment` (for the time axis).

### `plot_service_utilisation(service, env, ax=None)`

Bar + line combo — bar shows per-channel utilisation, line shows the
aggregate running mean over time. Handy for multi-channel `Service`.

### `plot_mc_fan(mc_result, percentiles=(5, 25, 50, 75, 95), ax=None)`

Given an `MCResult` where each sample is a time series, draws shaded
percentile bands around the median. The canonical "fan chart" used in
stochastic finance and epidemiology.

### `plot_warehouse_stock(warehouse, ax=None)`

Multi-line plot of stock levels per SKU over time, with dashed
horizontal lines at the reorder points.

### `plot_agent_path(agent, graph=None, ax=None)`

For grid-like graphs, renders the graph as a light grid + the agent's
traversal as a coloured path with task-completion markers.

---

## What `simweave.viz` will **not** do

- **No live/streaming dashboards.** That belongs in a separate
  `simweave-dashboard` package or a notebook pattern.
- **No 3-D rendering.** Engineering users sometimes want 3-D trajectories;
  keep that on the roadmap but not in first-cut.
- **No non-matplotlib backends.** Intentional to keep scope small. If
  and when plotly is needed, ship a parallel `simweave.viz.plotly`
  subpackage and keep the function signatures identical so swapping is
  trivial.
- **No custom "simweave theme".** Use matplotlib defaults; users have
  their own styles.

---

## Open questions for Stuart

1. Do you want the first cut to target **journal-style static figures**
   (good for reports / PDFs) or **notebook-interactive** (inline, with
   the ability to zoom)? Matplotlib does both, but default sizing and
   DPI differ.
2. For `plot_mc_fan`, should it accept a **raw `(n_runs, n_time)`
   ndarray** as well as an `MCResult`? (Likely yes — makes testing
   easier and lets users bring their own Monte Carlo.)
3. Any specific plot you want that I haven't listed? The current list
   reflects the demos in `demos/`. EdgeWeave may have its own "must
   show" plots worth codifying here.

---

## Rough schedule (best-endeavours, not a commitment)

Following the pattern we've been using:

- **0.2**: `simweave.currency` per `CURRENCY_DESIGN.md`.
- **0.3**: `simweave.viz` first-cut (state trajectories, phase portrait,
  queue length, utilisation, MC fan, warehouse stock, agent path).
- **0.4+**: optional plotly backend, optional 3-D, revisit whether a
  split-out package makes sense by then.
