# Visualisation

Plotly-based plot helpers, theme registry, and time-series recorders.
Requires the `[viz]` extra.

```bash
pip install "simweave[viz]"
```

## Plot helpers

Every helper returns a `plotly.graph_objects.Figure`. Each accepts an
optional `theme=` and `title=`.

| Helper                          | Input                                          |
| ------------------------------- | ---------------------------------------------- |
| `plot_state_trajectories`       | `SimulationResult`                             |
| `plot_phase_portrait`           | `SimulationResult` with ≥2 state channels      |
| `plot_queue_length`             | `QueueLengthRecorder`                          |
| `plot_service_utilisation`      | `ServiceUtilisationRecorder`                   |
| `plot_warehouse_stock`          | `WarehouseStockRecorder`                       |
| `plot_mc_fan`                   | `MCResult`, ndarray, or `(times, samples)`     |
| `plot_agent_path`               | `Agent` (with optional `graph=`)               |

## Recorders

Recorders are `Entity` subclasses. Snapshot at registration *and* at
the end of every tick. Register them after the entity they observe:

```python
qrec = sw.QueueLengthRecorder(svc)
env.register(svc)
env.register(qrec)            # ticks after svc
```

## Themes

```python
sw.available_themes()         # ['light', 'dark', 'presentation', 'minimal']
sw.set_default_theme("dark")
```

Brand themes via `register_theme`:

```python
sw.register_theme(
    "edgeweave",
    template="plotly_white",
    palette=["#1c5d99", "#f49d37", "#0b132b", "#5fad56"],
    layout_overrides={"font": {"family": "Inter, system-ui"}},
)
sw.set_default_theme("edgeweave")
```

## EdgeWeave consumption

Every figure round-trips through JSON cleanly:

```python
import json
fig = sw.plot_state_trajectories(res)
parsed = json.loads(fig.to_json())
assert "data" in parsed and "layout" in parsed
```

## Gallery

Live output of every helper, regenerated on each docs build:

<iframe src="../../embeds/msd_states.html"
        width="100%" height="480" frameborder="0"
        loading="lazy"
        title="State trajectories"></iframe>

<iframe src="../../embeds/msd_phase.html"
        width="100%" height="480" frameborder="0"
        loading="lazy"
        title="Phase portrait"></iframe>

<iframe src="../../embeds/queue_length.html"
        width="100%" height="420" frameborder="0"
        loading="lazy"
        title="Queue length"></iframe>

<iframe src="../../embeds/service_util.html"
        width="100%" height="520" frameborder="0"
        loading="lazy"
        title="Service utilisation"></iframe>

<iframe src="../../embeds/warehouse_stock.html"
        width="100%" height="500" frameborder="0"
        loading="lazy"
        title="Warehouse stock"></iframe>

<iframe src="../../embeds/mc_fan.html"
        width="100%" height="500" frameborder="0"
        loading="lazy"
        title="Monte Carlo fan"></iframe>

<iframe src="../../embeds/agent_path.html"
        width="100%" height="560" frameborder="0"
        loading="lazy"
        title="Agent path"></iframe>

## End-to-end demo

`demos/14_viz_tour.py` exercises every helper, writes one HTML per
figure under `demos/viz_out/`, and produces an `index.html`.

## API

::: simweave.