# Design note: `simweave.viz`

Status: **implemented in 0.3 (Plotly-first; matplotlib not in scope)**.
Original purpose: give users one-liner plotting helpers that turn the standard
simulation outputs — `SimulationResult`, `MCResult`, queue/service histories,
agent traversal, `Warehouse` stock trajectories — into the plots a simulation
engineer reaches for most often, **and** make every figure JSON-serialisable
so EdgeWeave (the JS frontend) can consume them directly via `fig.to_json()`.

---

## Extras vs separate library — decision

Shipped as a submodule under a `simweave[viz]` extra. The extra pulls
`plotly>=5.18`. matplotlib remains available via the older `simweave[plot]`
extra but no helpers in `simweave.viz` depend on it.

The triggers for splitting `simweave-viz` into its own package (multiple
backends, dashboard runtime, dedicated viz team, heavy deps like
pandas/bokeh/panel) still do not apply.

---

## Shipped surface (`simweave.viz`)

Convention: every helper **returns a `plotly.graph_objects.Figure`**. The
caller can mutate, restyle, or `fig.to_json()` it. All helpers accept an
optional `theme=` argument (string name of a registered theme) and an
optional `title=` argument.

Re-exported from the top-level `simweave` namespace as well, so:

```python
import simweave as sw

fig = sw.plot_state_trajectories(res, title="MSD")
fig.write_html("msd.html", include_plotlyjs="cdn")
blob = fig.to_json()                    # EdgeWeave consumption path
```

Public names exposed by `simweave.viz` (and re-exported on `simweave`):

| Name | Kind | Notes |
| ---- | ---- | ----- |
| `Theme` | dataclass | Frozen record: name + plotly template + palette + layout overrides |
| `apply_theme(fig, theme)` | helper | Mutates a figure in place to honour a theme |
| `available_themes()` | helper | Lists registered theme names |
| `get_theme(name)` / `get_default_theme()` | helpers | Lookup |
| `set_default_theme(name)` | helper | Switch the default for new figures |
| `register_theme(name, template, palette, layout_overrides, overwrite=False)` | helper | Add a brand theme |
| `QueueLengthRecorder(queue)` | Entity | Time-samples `len(queue)` each tick |
| `ServiceUtilisationRecorder(service)` | Entity | Per-channel busy_time + aggregate util |
| `WarehouseStockRecorder(warehouse)` | Entity | Per-SKU stock vector each tick |
| `plot_state_trajectories(result, channels=None, theme=None, title=None)` | plot | Continuous state vs time |
| `plot_phase_portrait(result, x_idx=0, y_idx=1, theme=None, title=None)` | plot | 2-state phase plane with start/end markers |
| `plot_queue_length(recorder, theme=None, title=None)` | plot | Step plot (`line.shape="hv"`) |
| `plot_service_utilisation(recorder, theme=None, title=None)` | plot | Aggregate util on y1, per-channel busy_time on y2 |
| `plot_warehouse_stock(recorder, sku_indices=None, theme=None, title=None, show_reorder_points=True)` | plot | Per-SKU lines + dashed reorder horizontals |
| `plot_mc_fan(mc_or_array, times=None, percentiles=(5,25,50,75,95), theme=None, title=None, show_mean=True)` | plot | Percentile fan chart; accepts `MCResult`, 2-D ndarray, or `(times, samples)` tuple |
| `plot_agent_path(agent, graph=None, show_graph=True, theme=None, title=None)` | plot | Agent traversal over a 2-D graph |
| `have_plotly()` | helper | Boolean probe; safe to call without the extra |

### Theme system

Built-in themes:

- `light` (default) — `plotly_white` template + Okabe-Ito palette
- `dark` — `plotly_dark` template + Okabe-Ito palette
- `presentation` — `presentation` template (large fonts)
- `minimal` — `simple_white` template + greyscale palette

Brand themes are added at import time without forking:

```python
import simweave as sw
sw.register_theme(
    "edgeweave",
    template="plotly_white",
    palette=["#1c5d99", "#f49d37", "#0b132b", "#5fad56"],
    layout_overrides={"font": {"family": "Inter, system-ui"}},
)
sw.set_default_theme("edgeweave")
```

### Recorders

Recorders are `Entity` subclasses. They snapshot at registration **and** at
the end of every tick, so the resulting time vector starts at `env.clock.start`.
Recommended: register the recorder *after* the entity it observes, so the
recorder's tick runs after the recorded entity has advanced.

```python
env.register(svc)
env.register(qrec)            # records svc's queue
```

### EdgeWeave consumption

Every figure round-trips through JSON cleanly:

```python
import json
blob = fig.to_json()                 # plotly's own serialiser
parsed = json.loads(blob)
assert "data" in parsed and "layout" in parsed
```

`tests/test_viz.py` exercises this for all seven helpers.

---

## What `simweave.viz` does **not** do

- **No live/streaming dashboards.** Belongs in a separate
  `simweave-dashboard` package or notebook pattern.
- **No 3-D rendering.** On the roadmap but not in first-cut.
- **No matplotlib backend.** First-cut Plotly-only, both for interactivity
  and because EdgeWeave needs a single serialisation format.
- **No automatic registration of recorders.** Users explicitly construct
  and register recorders so simulations don't carry a cost they didn't ask
  for.

---

## Resolved open questions

1. **Static vs interactive?** Interactive (Plotly), because EdgeWeave consumes
   the JSON. Static export remains available via `fig.write_image(...)`
   if `kaleido` is installed.
2. **`plot_mc_fan` input flexibility?** Yes. Accepts an `MCResult`, a raw
   2-D ndarray of shape `(n_runs, n_time)`, or a `(times, samples)` tuple.
3. **Plot list completeness?** The seven helpers cover the demos in
   `demos/` plus the agent path. Additional plot types (utilisation
   heatmaps, multi-warehouse stock comparisons, MC scatter matrices) are
   easy follow-ups but not in 0.3.
4. **Theming?** Decided in favour of a small `Theme` registry over relying
   on global Plotly templates, so brand colours can be wired in one call
   without touching `plotly.io.templates`.

---

## Implementation notes

Source layout (all under `src/simweave/viz/`):

- `_plotly.py` — lazy import of `plotly.graph_objects` with friendly
  `ImportError` (`pip install simweave[viz]`). `have_plotly()` is a
  boolean probe that never raises.
- `themes.py` — `Theme` dataclass, registry, `apply_theme(fig, theme)`,
  built-in themes, default-theme accessors.
- `recorders.py` — `_Recorder` base + `QueueLengthRecorder`,
  `ServiceUtilisationRecorder`, `WarehouseStockRecorder`. All
  pure-numpy, no plotly dependency.
- `plots.py` — the seven plot helpers. Each calls `require_go()` to
  obtain the plotly module lazily, builds a `Figure`, applies the
  selected theme, and returns it.
- `__init__.py` — explicit re-exports.

Top-level `simweave/__init__.py` re-exports every viz public name and adds
them to `__all__`. Importing `simweave` does **not** import plotly; only
the plot helpers do, and they raise a clear install hint if the extra
isn't present.

`pyproject.toml` carries `viz = ["plotly>=5.18"]`; `dev` includes plotly
so `tests/test_viz.py` runs in CI. Tests gate plotly-dependent assertions
behind `pytest.importorskip("plotly")` so the no-extras test path stays
green.

`demos/14_viz_tour.py` produces one HTML per helper plus an `index.html`
under `demos/viz_out/`. It also confirms a JSON round-trip for the
EdgeWeave consumption path.

A reusable Claude skill describing the public surface lives under
`claude_skill/SKILL.md` (installed locally to `~/.claude/skills/simweave/`).

---

## Schedule (delivered)

- **0.2**: `simweave.currency` per `CURRENCY_DESIGN.md` — shipped.
- **0.3**: `simweave.viz` first-cut — **shipped on `feature/viz`** (state
  trajectories, phase portrait, queue length, service utilisation,
  warehouse stock, Monte Carlo fan chart, agent path; theme registry; three
  recorders; demo tour; tests; Claude skill).

Possible 0.4 follow-ups (not committed): static-image export helper,
utilisation heatmap, multi-warehouse comparison plot, optional matplotlib
backend behind `simweave.viz.mpl`.
