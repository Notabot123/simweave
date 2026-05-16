"""Plot helpers for simweave.

Every helper:

* Lazy-imports plotly so the core package stays cheap.
* Returns a ``plotly.graph_objects.Figure`` so EdgeWeave (or any other
  JS frontend) can serialise via ``fig.to_json()`` and render natively.
* Applies the current default theme unless ``theme=`` is passed.

All time / value arrays are accepted as plain numpy or Python lists; the
helpers do not require a specific simweave object as long as the inputs
have the documented shape.

Time-axis helpers (``plot_queue_length``, ``plot_service_utilisation``,
``plot_warehouse_stock``, ``plot_fleet_availability``, ``plot_mc_fan``)
accept an optional ``time_axis`` keyword argument.  Pass a
:class:`~simweave.core.time_axis.SimTimeAxis` instance to replace the
numeric x-axis with real-world calendar dates.

Public API
----------
* :func:`plot_state_trajectories` -- continuous SimulationResult.
* :func:`plot_phase_portrait` -- 2-D phase portrait of a SimulationResult.
* :func:`plot_queue_length` -- queue length over time from a recorder.
* :func:`plot_service_utilisation` -- per-channel busy time + aggregate
  utilisation from a recorder.
* :func:`plot_warehouse_stock` -- per-SKU stock levels from a recorder.
* :func:`plot_mc_fan` -- percentile fan chart for a Monte Carlo trajectory
  ensemble (accepts ``MCResult``, raw 2D array, or ``(times, samples)``).
* :func:`plot_agent_path` -- agent traversal over a 2-D graph.
* :func:`plot_fleet_availability` -- stacked area chart of fleet availability.
* :func:`plot_sensitivity_surface` -- 3-D surface / heatmap / grouped bar
  for a sensitivity sweep result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from simweave.viz import _plotly
from simweave.viz.themes import apply_theme
from simweave.supplychain.warehouse import Warehouse
from simweave.supplychain.optimization import pareto_sweep

if TYPE_CHECKING:
    from simweave.core.time_axis import SimTimeAxis


def _apply_time_axis(fig: Any, time_axis: "SimTimeAxis | None") -> Any:
    """Apply a SimTimeAxis to a figure if provided; otherwise no-op."""
    if time_axis is not None:
        time_axis.apply_to_figure(fig, axis="x", title="date")
    return fig


# --------------------------------------------------------------------------- #
# Continuous: state trajectories + phase portrait                             #
# --------------------------------------------------------------------------- #


def plot_state_trajectories(
    result: Any,
    channels: Sequence[int] | None = None,
    *,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Line plot of each state channel vs ``result.time``.

    Parameters
    ----------
    result:
        A :class:`~simweave.continuous.solver.SimulationResult` or any
        object exposing ``.time``, ``.state`` (shape ``(T, n)``) and
        ``.state_labels``.
    channels:
        Optional indices of state channels to include. Default: all.
    theme:
        Theme name. Default: the global default (see
        :func:`simweave.viz.set_default_theme`).
    title:
        Optional figure title.
    """
    go = _plotly.require_go()
    time = np.asarray(result.time)
    state = np.asarray(result.state)
    labels = list(getattr(result, "state_labels", ()) or [
        f"x{i}" for i in range(state.shape[1])
    ])
    idxs = list(channels) if channels is not None else list(range(state.shape[1]))

    fig = go.Figure()
    for i in idxs:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=state[:, i],
                mode="lines",
                name=labels[i] if i < len(labels) else f"x{i}",
            )
        )
    sys_name = getattr(result, "system_name", "system")
    fig.update_layout(
        title=title or f"{sys_name} -- state trajectories",
        xaxis_title="time",
        yaxis_title="state",
        legend_title_text="channel",
    )
    return apply_theme(fig, theme)


def plot_phase_portrait(
    result: Any,
    x_idx: int = 0,
    y_idx: int = 1,
    *,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Plot ``state[:, x_idx]`` vs ``state[:, y_idx]`` with a start marker."""
    go = _plotly.require_go()
    state = np.asarray(result.state)
    if state.shape[1] <= max(x_idx, y_idx):
        raise IndexError(
            f"phase portrait needs at least {max(x_idx, y_idx) + 1} state channels; "
            f"got {state.shape[1]}."
        )
    labels = list(getattr(result, "state_labels", ()) or [
        f"x{i}" for i in range(state.shape[1])
    ])
    xs = state[:, x_idx]
    ys = state[:, y_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="trajectory",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xs[0]],
            y=[ys[0]],
            mode="markers",
            marker={"size": 10, "symbol": "circle"},
            name="start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xs[-1]],
            y=[ys[-1]],
            mode="markers",
            marker={"size": 10, "symbol": "x"},
            name="end",
        )
    )
    sys_name = getattr(result, "system_name", "system")
    fig.update_layout(
        title=title or f"{sys_name} -- phase portrait",
        xaxis_title=labels[x_idx] if x_idx < len(labels) else f"x{x_idx}",
        yaxis_title=labels[y_idx] if y_idx < len(labels) else f"y{y_idx}",
    )
    return apply_theme(fig, theme)


# --------------------------------------------------------------------------- #
# Discrete: queue length and service utilisation (via recorders)              #
# --------------------------------------------------------------------------- #


def plot_queue_length(
    recorder: Any,
    *,
    time_axis: "SimTimeAxis | None" = None,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Step plot of queue length over time.

    Parameters
    ----------
    recorder:
        A :class:`~simweave.viz.recorders.QueueLengthRecorder` (or anything
        exposing ``.times`` and ``.lengths``).
    time_axis:
        Optional :class:`~simweave.core.time_axis.SimTimeAxis`.  When
        supplied, the x-axis shows calendar dates instead of numeric ticks.
    """
    go = _plotly.require_go()
    times = np.asarray(recorder.times)
    lengths = np.asarray(recorder.lengths)
    qname = getattr(recorder.queue, "name", "queue") if hasattr(recorder, "queue") else "queue"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=lengths,
            mode="lines",
            line={"shape": "hv"},
            name=qname,
        )
    )
    fig.update_layout(
        title=title or f"queue length: {qname}",
        xaxis_title="time",
        yaxis_title="length",
    )
    return apply_theme(_apply_time_axis(fig, time_axis), theme)


def plot_service_utilisation(
    recorder: Any,
    *,
    time_axis: "SimTimeAxis | None" = None,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Aggregate utilisation line + per-channel busy-time lines.

    Parameters
    ----------
    recorder:
        A :class:`~simweave.viz.recorders.ServiceUtilisationRecorder`.
    time_axis:
        Optional :class:`~simweave.core.time_axis.SimTimeAxis` for calendar
        date x-axis labels.
    """
    go = _plotly.require_go()
    times = np.asarray(recorder.times)
    util = np.asarray(recorder.utilisation)
    busy = np.asarray(recorder.busy_time)  # (T, n_channels)
    sname = (
        getattr(recorder.service, "name", "service")
        if hasattr(recorder, "service")
        else "service"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=util,
            mode="lines",
            name="aggregate utilisation",
            line={"width": 3},
        )
    )
    if busy.ndim == 2:
        for ch in range(busy.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=busy[:, ch],
                    mode="lines",
                    name=f"channel {ch} busy_time",
                    line={"dash": "dot"},
                    yaxis="y2",
                )
            )
    fig.update_layout(
        title=title or f"service utilisation: {sname}",
        xaxis_title="time",
        yaxis={"title": "aggregate utilisation", "range": [0, 1]},
        yaxis2={
            "title": "cumulative busy time per channel",
            "overlaying": "y",
            "side": "right",
        },
        legend={"orientation": "h", "y": -0.2},
    )
    return apply_theme(_apply_time_axis(fig, time_axis), theme)


# --------------------------------------------------------------------------- #
# Supply chain: warehouse stock                                               #
# --------------------------------------------------------------------------- #


def plot_warehouse_stock(
    recorder: Any,
    sku_indices: Sequence[int] | None = None,
    *,
    time_axis: "SimTimeAxis | None" = None,
    theme: str | None = None,
    title: str | None = None,
    show_reorder_points: bool = True,
) -> Any:
    """Per-SKU stock-level lines with optional reorder-point dashes.

    Parameters
    ----------
    recorder:
        A :class:`~simweave.viz.recorders.WarehouseStockRecorder`.
    sku_indices:
        Optional iterable of SKU indices to include. Default: all.
    time_axis:
        Optional :class:`~simweave.core.time_axis.SimTimeAxis` for calendar
        date x-axis labels.
    show_reorder_points:
        If True, draws a horizontal dashed line at each SKU's reorder
        point in the same colour family as its stock trace.
    """
    go = _plotly.require_go()
    times = np.asarray(recorder.times)
    stock = np.asarray(recorder.stock)  # (T, n_skus)
    names = list(recorder.sku_names)
    rop = np.asarray(recorder.reorder_points)

    idxs = list(sku_indices) if sku_indices is not None else list(range(stock.shape[1]))

    fig = go.Figure()
    for i in idxs:
        sku = names[i] if i < len(names) else f"sku{i}"
        fig.add_trace(
            go.Scatter(
                x=times,
                y=stock[:, i],
                mode="lines",
                name=sku,
            )
        )
        if show_reorder_points and i < rop.size:
            fig.add_trace(
                go.Scatter(
                    x=[times[0], times[-1]],
                    y=[rop[i], rop[i]],
                    mode="lines",
                    name=f"{sku} reorder",
                    line={"dash": "dash", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    wname = (
        getattr(recorder.warehouse, "name", "warehouse")
        if hasattr(recorder, "warehouse")
        else "warehouse"
    )
    fig.update_layout(
        title=title or f"warehouse stock: {wname}",
        xaxis_title="time",
        yaxis_title="stock level",
        legend_title_text="SKU",
    )
    return apply_theme(_apply_time_axis(fig, time_axis), theme)

# --------------------------------------------------------------------------- #
# Inventory optimisation: pareto sweep                                        #
# --------------------------------------------------------------------------- #


def plot_pareto_sweep(
    warehouse: Warehouse,
    availability_range: np.ndarray | None = None,
    *,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Plot cost vs availability for both heuristic and DE optimisation.

    Parameters
    ----------
    warehouse:
        The Warehouse instance to optimise.
    availability_range:
        Optional array of availability targets. Default: 0.1 → 0.95 in steps of 0.05.
    theme:
        Optional Plotly theme name.
    title:
        Optional plot title.
    """
    go = _plotly.require_go()

    sweep = pareto_sweep(warehouse, availability_range)
    a = sweep["availability"]
    c_de = sweep["cost_cost_optimal"]
    c_p = sweep["cost_poisson"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=a,
            y=c_de,
            mode="lines+markers",
            name="cost-optimal (DE)",
            line={"width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=a,
            y=c_p,
            mode="lines+markers",
            name="Poisson heuristic",
            line={"dash": "dash", "width": 2},
        )
    )

    wname = getattr(warehouse, "name", "warehouse")

    fig.update_layout(
        title=title or f"cost vs availability: {wname}",
        xaxis_title="target availability",
        yaxis_title="total cost",
        legend_title_text="method",
    )

    return apply_theme(fig, theme)


# --------------------------------------------------------------------------- #
# Monte Carlo: percentile fan                                                 #
# --------------------------------------------------------------------------- #


def _coerce_mc_input(
    obj: Any,
    times: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, samples)`` where samples is ``(n_runs, n_time)``.

    Accepts:
      * ``MCResult`` whose ``.samples`` is shape ``(n_runs, n_time)``
      * raw 2-D ndarray ``(n_runs, n_time)``
      * a ``(times, samples)`` tuple
    """
    if isinstance(obj, tuple) and len(obj) == 2:
        t_in, s_in = obj
        return np.asarray(t_in, dtype=float), np.asarray(s_in, dtype=float)

    samples_attr = getattr(obj, "samples", None)
    samples = np.asarray(samples_attr if samples_attr is not None else obj, dtype=float)
    if samples.ndim != 2:
        raise ValueError(
            "plot_mc_fan expects an (n_runs, n_time) 2-D array of trajectories; "
            f"got shape {samples.shape}."
        )
    if times is None:
        t_arr = np.arange(samples.shape[1], dtype=float)
    else:
        t_arr = np.asarray(times, dtype=float)
        if t_arr.size != samples.shape[1]:
            raise ValueError(
                f"times length ({t_arr.size}) does not match sample width "
                f"({samples.shape[1]})."
            )
    return t_arr, samples


def plot_mc_fan(
    mc_or_array: Any,
    times: Sequence[float] | np.ndarray | None = None,
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
    *,
    time_axis: "SimTimeAxis | None" = None,
    theme: str | None = None,
    title: str | None = None,
    show_mean: bool = True,
) -> Any:
    """Percentile fan chart for a Monte Carlo trajectory ensemble.

    Parameters
    ----------
    mc_or_array:
        ``MCResult`` (with 2-D ``.samples``), raw ``(n_runs, n_time)``
        ndarray, or a ``(times, samples)`` tuple.
    times:
        Optional time axis. Ignored if ``mc_or_array`` is a tuple.
        Defaults to ``arange(n_time)`` if not supplied.
    percentiles:
        Percentiles to draw. The median (or 50th percentile if present)
        is rendered as a solid line; the rest form symmetric shaded bands
        from outside-in.
    show_mean:
        If True, overlays the per-time mean as a thin dashed line.
    """
    go = _plotly.require_go()
    t, samples = _coerce_mc_input(mc_or_array, times)

    pct = sorted(float(p) for p in percentiles)
    if not pct:
        raise ValueError("percentiles must be non-empty.")
    qs = np.percentile(samples, pct, axis=0)  # shape (len(pct), n_time)

    # Pair outside-in for shading: (lo, hi) -> band; if odd count, the middle one
    # becomes the median line.
    bands: list[tuple[int, int]] = []
    i, j = 0, len(pct) - 1
    while i < j:
        bands.append((i, j))
        i += 1
        j -= 1
    median_idx = i if (i == j) else None

    fig = go.Figure()
    n_bands = len(bands)
    for k, (lo, hi) in enumerate(bands):
        # Outer band fades the most; inner band darkest.
        opacity = 0.15 + 0.45 * (k / max(n_bands - 1, 1))
        # Lower trace (invisible line, will be filled to)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=qs[lo],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=qs[hi],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=f"rgba(31, 119, 180, {opacity:.3f})",
                name=f"P{pct[lo]:g}-P{pct[hi]:g}",
                hoverinfo="skip",
            )
        )
    if median_idx is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=qs[median_idx],
                mode="lines",
                line={"width": 2},
                name=f"P{pct[median_idx]:g} (median)",
            )
        )
    if show_mean:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=samples.mean(axis=0),
                mode="lines",
                line={"dash": "dash", "width": 1.5},
                name="mean",
            )
        )

    scenario = (
        getattr(mc_or_array, "scenario_name", None)
        if not isinstance(mc_or_array, (tuple, np.ndarray))
        else None
    )
    fig.update_layout(
        title=title
        or (f"Monte Carlo fan: {scenario}" if scenario else "Monte Carlo fan"),
        xaxis_title="time",
        yaxis_title="value",
    )
    return apply_theme(_apply_time_axis(fig, time_axis), theme)


# --------------------------------------------------------------------------- #
# Agents: traversal of a 2-D graph                                            #
# --------------------------------------------------------------------------- #


def _node_xy(node: Any, graph: Any | None) -> tuple[float, float] | None:
    """Best-effort extraction of an (x, y) for a node.

    Handles:
    * tuple ``(r, c)`` -> ``(c, r)`` (col=x, row=y; row inverted not required
      here since we let plotly pick the axis direction)
    * networkx graph with ``nodes[n]['pos']`` -> use that
    * anything else -> None
    """
    if isinstance(node, tuple) and len(node) == 2:
        try:
            return float(node[1]), float(node[0])
        except (TypeError, ValueError):
            pass
    if graph is not None and hasattr(graph, "nodes"):
        try:
            data = graph.nodes[node]  # networkx-style
            pos = data.get("pos")
            if pos is not None and len(pos) >= 2:
                return float(pos[0]), float(pos[1])
        except Exception:
            pass
    return None


def plot_agent_path(
    agent: Any,
    graph: Any | None = None,
    *,
    show_graph: bool = True,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Render an agent's traversal over a 2-D graph.

    Parameters
    ----------
    agent:
        A :class:`~simweave.agents.agent.Agent` (or any object exposing
        ``.history`` of ``(t, node)`` pairs and ``.name``).
    graph:
        Optional graph to draw underneath the path. Defaults to
        ``agent.graph``. Pass ``show_graph=False`` to skip drawing it.
    show_graph:
        If True, draws faint lines for every edge of the graph.
    """
    go = _plotly.require_go()
    g = graph if graph is not None else getattr(agent, "graph", None)

    history: Iterable[tuple[float, Any]] = getattr(agent, "history", []) or []
    times: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    nodes: list[Any] = []
    for t, node in history:
        xy = _node_xy(node, g)
        if xy is None:
            continue
        times.append(float(t))
        xs.append(xy[0])
        ys.append(xy[1])
        nodes.append(node)

    fig = go.Figure()

    if show_graph and g is not None:
        edge_x: list[Any] = []
        edge_y: list[Any] = []
        edges_iter = (
            g.edges() if hasattr(g, "edges") and callable(getattr(g, "edges")) else []
        )
        for edge in edges_iter:
            # simweave Graph.edges() yields (u, v, data); networkx also (u, v) or 3-tup.
            u, v = edge[0], edge[1]
            uxy = _node_xy(u, g)
            vxy = _node_xy(v, g)
            if uxy is None or vxy is None:
                continue
            edge_x.extend([uxy[0], vxy[0], None])
            edge_y.extend([uxy[1], vxy[1], None])
        if edge_x:
            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line={"color": "rgba(150,150,150,0.4)", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                    name="graph",
                )
            )

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                marker={"size": 7},
                line={"width": 2},
                name=getattr(agent, "name", "agent"),
                hovertext=[f"t={tt:.2f}\n{n}" for tt, n in zip(times, nodes)],
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[xs[0]],
                y=[ys[0]],
                mode="markers",
                marker={"size": 12, "symbol": "circle"},
                name="start",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[xs[-1]],
                y=[ys[-1]],
                mode="markers",
                marker={"size": 12, "symbol": "x"},
                name="end",
                showlegend=True,
            )
        )

    aname = getattr(agent, "name", "agent")
    fig.update_layout(
        title=title or f"agent path: {aname}",
        xaxis_title="x",
        yaxis_title="y",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    return apply_theme(fig, theme)


# --------------------------------------------------------------------------- #
# Reliability: fleet availability stacked area                                #
# --------------------------------------------------------------------------- #


def plot_fleet_availability(
    recorder: Any,
    *,
    normalize: bool = False,
    time_axis: "SimTimeAxis | None" = None,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Stacked area chart of fleet operational availability over time.

    The three stacked layers are (bottom to top):

    * **awaiting part** -- entities grounded waiting for spare stock (red).
    * **in repair** -- parts obtained, repair in progress (amber).
    * **operational** -- fully mission-capable (green).

    Parameters
    ----------
    recorder:
        A :class:`~simweave.reliability.fleet.FleetAvailabilityRecorder`
        (or any object exposing ``.times``, ``.operational``,
        ``.in_repair``, ``.awaiting_part``, and ``.fleet``).
    normalize:
        If ``True``, express each band as a fraction of fleet size (0–1)
        rather than an absolute vehicle count.
    theme:
        Theme name.
    title:
        Optional figure title.
    """
    go = _plotly.require_go()
    times = np.asarray(recorder.times)
    op = np.asarray(recorder.operational, dtype=float)
    ir = np.asarray(recorder.in_repair, dtype=float)
    ap = np.asarray(recorder.awaiting_part, dtype=float)

    fleet_name = getattr(getattr(recorder, "fleet", None), "name", "fleet")
    fleet_size = getattr(getattr(recorder, "fleet", None), "size", 1) or 1

    if normalize:
        op = op / fleet_size
        ir = ir / fleet_size
        ap = ap / fleet_size
        ylabel = "fraction of fleet"
    else:
        ylabel = "vehicles"

    fig = go.Figure()
    # Stack order: awaiting_part at bottom, then in_repair, then operational.
    fig.add_trace(
        go.Scatter(
            x=times,
            y=ap,
            mode="lines",
            stackgroup="one",
            name="awaiting part",
            line={"color": "rgba(180,30,30,0.9)", "width": 0.5},
            fillcolor="rgba(214,39,40,0.65)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=ir,
            mode="lines",
            stackgroup="one",
            name="in repair",
            line={"color": "rgba(200,100,0,0.9)", "width": 0.5},
            fillcolor="rgba(255,127,14,0.65)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=op,
            mode="lines",
            stackgroup="one",
            name="operational",
            line={"color": "rgba(30,140,30,0.9)", "width": 0.5},
            fillcolor="rgba(44,160,44,0.65)",
        )
    )
    fig.update_layout(
        title=title or f"fleet availability: {fleet_name}",
        xaxis_title="time",
        yaxis_title=ylabel,
        legend={"orientation": "h", "y": -0.2},
    )
    return apply_theme(_apply_time_axis(fig, time_axis), theme)


# --------------------------------------------------------------------------- #
# Reliability: sensitivity analysis surface / heatmap                         #
# --------------------------------------------------------------------------- #


def plot_sensitivity_surface(
    result: Any,
    *,
    chart_type: str = "surface",
    show_std: bool = False,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """3-D surface or grouped bar chart for a 2-D sensitivity sweep result.

    Parameters
    ----------
    result:
        A :class:`~simweave.reliability.sensitivity.SweepResult` from a 2-D
        sweep (``result.is_2d`` must be ``True``).
    chart_type:
        ``"surface"`` (default) -- smooth 3-D surface using ``go.Surface``.
        ``"bar"`` -- grouped 2-D bar chart (p1 values as groups, p2 as
        series), useful when parameter values are discrete labels.
        ``"heatmap"`` -- 2-D colour map, ideal for dense grids.
    show_std:
        If ``True`` and ``chart_type="bar"``, draws error bars representing
        ±1 standard deviation across Monte Carlo replicates.
    theme:
        Theme name.
    title:
        Optional figure title.
    """
    go = _plotly.require_go()

    p1 = np.asarray(result.param1_values)
    metric = np.asarray(result.metric_mean)
    std = np.asarray(result.metric_std)
    p1_name = result.param1_name
    metric_name = result.metric_name

    # 1-D fallback: line plot with optional error band.
    if not result.is_2d:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=p1,
                y=metric,
                mode="lines+markers",
                name=metric_name,
                error_y=(
                    {"type": "data", "array": std.tolist(), "visible": True}
                    if show_std
                    else None
                ),
            )
        )
        fig.update_layout(
            title=title or f"sensitivity: {metric_name} vs {p1_name}",
            xaxis_title=p1_name,
            yaxis_title=metric_name,
        )
        return apply_theme(fig, theme)

    # 2-D
    p2 = np.asarray(result.param2_values)
    p2_name = result.param2_name

    if chart_type == "surface":
        fig = go.Figure(
            go.Surface(
                x=p2,
                y=p1,
                z=metric,
                colorscale="Viridis",
                colorbar={"title": metric_name},
            )
        )
        fig.update_layout(
            title=title or f"sensitivity surface: {metric_name}",
            scene={
                "xaxis_title": p2_name,
                "yaxis_title": p1_name,
                "zaxis_title": metric_name,
            },
        )

    elif chart_type == "heatmap":
        fig = go.Figure(
            go.Heatmap(
                x=[str(v) for v in p2],
                y=[str(v) for v in p1],
                z=metric,
                colorscale="Viridis",
                colorbar={"title": metric_name},
                hoverongaps=False,
            )
        )
        fig.update_layout(
            title=title or f"sensitivity heatmap: {metric_name}",
            xaxis_title=p2_name,
            yaxis_title=p1_name,
        )

    else:  # "bar"
        fig = go.Figure()
        for j, v2 in enumerate(p2):
            ey = {"type": "data", "array": std[:, j].tolist(), "visible": True} if show_std else None
            fig.add_trace(
                go.Bar(
                    name=f"{p2_name}={v2:g}",
                    x=[f"{v:g}" for v in p1],
                    y=metric[:, j],
                    error_y=ey,
                )
            )
        fig.update_layout(
            title=title or f"sensitivity: {metric_name}",
            xaxis_title=p1_name,
            yaxis_title=metric_name,
            barmode="group",
        )

    return apply_theme(fig, theme)


# --------------------------------------------------------------------------- #
# Roads: occupancy and queue charts                                            #
# --------------------------------------------------------------------------- #


def plot_road_occupancy(
    recorder: Any,
    *,
    time_axis: "SimTimeAxis | None" = None,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Line chart of in-transit vehicle counts for a set of roads over time.

    Parameters
    ----------
    recorder:
        A :class:`~simweave.roads.recorder.RoadOccupancyRecorder` (or any
        object exposing ``.times``, ``.occupancy``, and ``.road_names``).
    time_axis:
        Optional :class:`~simweave.core.time_axis.SimTimeAxis` for calendar
        date x-axis labels.
    theme, title:
        Standard theme/title overrides.
    """
    go = _plotly.require_go()
    times = np.asarray(recorder.times)
    occ = np.asarray(recorder.occupancy)   # (n_samples, n_roads)
    road_names: Sequence[str] = recorder.road_names

    fig = go.Figure()
    for i, name in enumerate(road_names):
        col = occ[:, i] if occ.ndim == 2 else occ
        fig.add_trace(
            go.Scatter(
                x=times,
                y=col,
                mode="lines",
                name=name,
            )
        )
    fig.update_layout(
        title=title or "Road occupancy (vehicles in transit)",
        xaxis_title="time",
        yaxis_title="vehicles in transit",
        legend={"orientation": "h", "y": -0.2},
    )
    return apply_theme(_apply_time_axis(fig, time_axis), theme)


def plot_intersection_queues(
    recorder: Any,
    *,
    time_axis: "SimTimeAxis | None" = None,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Line chart of approach-queue lengths at one intersection over time.

    The total queue (solid line) and each per-approach breakdown (dashed
    lines) are plotted.

    Parameters
    ----------
    recorder:
        An :class:`~simweave.roads.recorder.IntersectionQueueRecorder` (or
        any object exposing ``.times``, ``.total_queued``, and
        ``.per_approach``).
    time_axis, theme, title:
        Standard overrides.
    """
    go = _plotly.require_go()
    times = np.asarray(recorder.times)
    total = np.asarray(recorder.total_queued, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=total,
            mode="lines",
            name="total queued",
            line={"width": 2},
        )
    )

    # Per-approach breakdown
    if recorder.per_approach:
        approach_names = list(recorder.per_approach[0].keys())
        for road_name in approach_names:
            vals = np.asarray(
                [d.get(road_name, 0) for d in recorder.per_approach], dtype=float
            )
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=vals,
                    mode="lines",
                    name=road_name,
                    line={"dash": "dot", "width": 1},
                )
            )

    fig.update_layout(
        title=title or "Intersection queue lengths",
        xaxis_title="time",
        yaxis_title="vehicles queued",
        legend={"orientation": "h", "y": -0.2},
    )
    return apply_theme(_apply_time_axis(fig, time_axis), theme)


def plot_fault_signals(
    result: Any,
    injector: Any,
    channels: Sequence[int] | None = None,
    *,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """State trajectories with shaded regions for degraded and failed phases.

    The plot overlays the simulated state channels with coloured background
    bands:

    * **green** — healthy (health index = 1).
    * **amber** — degrading (0 < health index < 1).
    * **red** — failed (health index = 0).

    Parameters
    ----------
    result:
        A :class:`~simweave.continuous.solver.SimulationResult` from a run
        that used a :class:`~simweave.faults.injector.FaultInjector`.
    injector:
        The :class:`~simweave.faults.injector.FaultInjector` used for the run
        (provides fault profile timings).
    channels:
        State channel indices to plot.  Default: all.
    theme, title:
        Standard overrides.
    """
    go = _plotly.require_go()
    time = np.asarray(result.time)
    state = np.asarray(result.state)
    labels = list(getattr(result, "state_labels", ()) or [
        f"x{i}" for i in range(state.shape[1])
    ])
    idxs = list(channels) if channels is not None else list(range(state.shape[1]))

    fig = go.Figure()

    # Shade degradation / failure windows
    t0, tf = float(time[0]), float(time[-1])
    for f in injector.faults:
        onset = max(f.profile.onset_time, t0)
        failure = min(f.profile.failure_time, tf)
        if onset < failure:
            fig.add_vrect(
                x0=onset, x1=failure,
                fillcolor="orange", opacity=0.12, layer="below", line_width=0,
                annotation_text=f"degrading ({f.profile.mode})",
                annotation_position="top left",
            )
        if failure < tf:
            fig.add_vrect(
                x0=failure, x1=tf,
                fillcolor="red", opacity=0.12, layer="below", line_width=0,
                annotation_text="failed",
                annotation_position="top left",
            )

    for i in idxs:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=state[:, i],
                mode="lines",
                name=labels[i] if i < len(labels) else f"x{i}",
            )
        )

    sys_name = getattr(result, "system_name", "system")
    fig.update_layout(
        title=title or f"{sys_name} — fault signals",
        xaxis_title="time",
        yaxis_title="state",
        legend_title_text="channel",
    )
    return apply_theme(fig, theme)


def plot_health_index(
    dataset: Any,
    *,
    show_rul: bool = True,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Health index (and optionally RUL) over simulation time.

    Parameters
    ----------
    dataset:
        A :class:`~simweave.faults.dataset.FaultDataset` (or any object
        exposing ``.time``, ``.health_index``, and ``.rul``).
    show_rul:
        If ``True`` (default), overlay remaining useful life on a secondary
        y-axis.
    theme, title:
        Standard overrides.
    """
    go = _plotly.require_go()
    time = np.asarray(dataset.time)
    hi = np.asarray(dataset.health_index)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=hi,
            mode="lines",
            name="health index",
            line={"color": "royalblue", "width": 2},
        )
    )

    if show_rul:
        rul = np.asarray(dataset.rul, dtype=float)
        rul_finite = np.where(np.isinf(rul), np.nan, rul)
        if not np.all(np.isnan(rul_finite)):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=rul_finite,
                    mode="lines",
                    name="RUL",
                    line={"color": "darkorange", "dash": "dash", "width": 1.5},
                    yaxis="y2",
                )
            )
            fig.update_layout(
                yaxis2={
                    "title": "remaining useful life",
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                }
            )

    fig.update_layout(
        title=title or "Health index and RUL",
        xaxis_title="time",
        yaxis={"title": "health index", "range": [-0.05, 1.05]},
        legend={"orientation": "h", "y": -0.2},
    )
    return apply_theme(fig, theme)


__all__ = [
    "plot_state_trajectories",
    "plot_phase_portrait",
    "plot_queue_length",
    "plot_service_utilisation",
    "plot_warehouse_stock",
    "plot_pareto_sweep",
    "plot_mc_fan",
    "plot_agent_path",
    "plot_fleet_availability",
    "plot_sensitivity_surface",
    "plot_road_occupancy",
    "plot_intersection_queues",
    "plot_fault_signals",
    "plot_health_index",
]
