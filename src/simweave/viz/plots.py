"""Plot helpers for simweave.

Every helper:

* Lazy-imports plotly so the core package stays cheap.
* Returns a ``plotly.graph_objects.Figure`` so EdgeWeave (or any other
  JS frontend) can serialise via ``fig.to_json()`` and render natively.
* Applies the current default theme unless ``theme=`` is passed.

All time / value arrays are accepted as plain numpy or Python lists; the
helpers do not require a specific simweave object as long as the inputs
have the documented shape.

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
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from simweave.viz import _plotly
from simweave.viz.themes import apply_theme


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
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Step plot of queue length over time.

    Parameters
    ----------
    recorder:
        A :class:`~simweave.viz.recorders.QueueLengthRecorder` (or anything
        exposing ``.times`` and ``.lengths``).
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
    return apply_theme(fig, theme)


def plot_service_utilisation(
    recorder: Any,
    *,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Aggregate utilisation line + per-channel busy-time lines.

    Parameters
    ----------
    recorder:
        A :class:`~simweave.viz.recorders.ServiceUtilisationRecorder`.
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
    return apply_theme(fig, theme)


# --------------------------------------------------------------------------- #
# Supply chain: warehouse stock                                               #
# --------------------------------------------------------------------------- #


def plot_warehouse_stock(
    recorder: Any,
    sku_indices: Sequence[int] | None = None,
    *,
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
    return apply_theme(fig, theme)


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


__all__ = [
    "plot_state_trajectories",
    "plot_phase_portrait",
    "plot_queue_length",
    "plot_service_utilisation",
    "plot_warehouse_stock",
    "plot_mc_fan",
    "plot_agent_path",
]
