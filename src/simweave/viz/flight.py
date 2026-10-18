"""Plot helpers for :mod:`simweave.flight` results.

* :func:`plot_trajectory_3d` -- the flown path in 3D, coloured by speed, with
  the reference path.
* :func:`plot_flight_summary` -- tracking error, tilt, rotor thrusts and power
  over time, with mission phases marked.
* :func:`plot_landing_dispersion` -- landing points from a Monte Carlo run
  with containment circles.

All accept a :class:`~simweave.flight.results.FlightResult` (or any object
exposing the same attributes), lazy-import plotly, and apply the current theme.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simweave.viz import _plotly
from simweave.viz.themes import apply_theme

_PHASE_SHADE = "rgba(128,128,128,0.10)"


def plot_trajectory_3d(
    flight: Any,
    *,
    show_reference: bool = True,
    color_by_speed: bool = True,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """3D path of the vehicle (Z up) with start and end markers.

    Parameters
    ----------
    flight:
        A :class:`~simweave.flight.results.FlightResult`.
    show_reference:
        Also draw the reference path as a dashed line.
    color_by_speed:
        Colour the flown path by speed (m/s); otherwise a single colour.
    """
    go = _plotly.require_go()
    pos = np.asarray(flight.position)
    fig = go.Figure()

    if show_reference:
        ref = np.asarray(flight.reference_position)
        fig.add_trace(
            go.Scatter3d(
                x=ref[:, 0], y=ref[:, 1], z=ref[:, 2], mode="lines",
                name="reference", line=dict(dash="dash", width=3),
                opacity=0.6,
            )
        )

    if color_by_speed:
        speed = np.linalg.norm(np.asarray(flight.velocity), axis=1)
        line = dict(
            width=6, color=speed, colorscale="Viridis",
            colorbar=dict(title="speed (m/s)", len=0.6, y=0.3),
        )
    else:
        line = dict(width=6)
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines",
            name="flown", line=line,
        )
    )
    if np.linalg.norm(pos[-1] - pos[0]) < 0.25:  # a round trip: one marker
        ends, labels = pos[:1], ["start / end"]
    else:
        ends, labels = pos[[0, -1]], ["start", "end"]
    fig.add_trace(
        go.Scatter3d(
            x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="markers+text",
            text=labels, textposition="top center",
            marker=dict(size=5), name="start / end", showlegend=False,
        )
    )
    fig.update_layout(
        title=title or "Flight path",
        scene=dict(
            aspectmode="data",
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
        ),
        legend=dict(x=0.0, y=1.0, title=dict(text="path")),
    )
    return apply_theme(fig, theme)


def plot_flight_summary(
    flight: Any,
    *,
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Four stacked panels over time: tracking error, tilt, rotor thrusts, power.

    If the flight's reference is a :class:`~simweave.flight.trajectory.Mission`
    its phases are shaded and labelled on the top panel. The rotor panel
    shows the thrust limit and hover thrust as reference lines.
    """
    go = _plotly.require_go()
    from plotly.subplots import make_subplots

    t = np.asarray(flight.time)
    veh = flight.vehicle
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(
            "Position error (m)", "Tilt (deg)", "Rotor thrust (N)", "Rotor power (W)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=t, y=flight.tracking_error(), mode="lines", name="error",
                   showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=np.degrees(flight.tilt()), mode="lines", name="tilt",
                   showlegend=False),
        row=2, col=1,
    )
    thrusts = np.asarray(flight.thrusts)
    for i in range(thrusts.shape[1]):
        fig.add_trace(
            go.Scatter(x=t, y=thrusts[:, i], mode="lines", name=f"rotor {i + 1}"),
            row=3, col=1,
        )
    fig.add_hline(y=veh.max_thrust, line_dash="dot", row=3, col=1,
                  annotation_text="max thrust", annotation_position="top left")
    fig.add_hline(y=veh.hover_thrust, line_dash="dot", row=3, col=1,
                  annotation_text="hover", annotation_position="bottom left")
    fig.add_trace(
        go.Scatter(x=t, y=flight.power(), mode="lines", name="power",
                   showlegend=False),
        row=4, col=1,
    )

    phases = getattr(flight.reference, "phases", None)
    if phases:
        for k, (name, t0, t1) in enumerate(phases):
            if k % 2 == 0:
                fig.add_vrect(x0=t0, x1=t1, fillcolor=_PHASE_SHADE, line_width=0,
                              row="all", col=1)
            fig.add_annotation(
                x=(t0 + t1) / 2, y=1.075, xref="x", yref="paper", text=name,
                showarrow=False, yanchor="bottom", font=dict(size=10),
            )

    fig.update_xaxes(title_text="time (s)", row=4, col=1)
    fig.update_layout(
        title=dict(text=title or "Flight summary", y=0.985, yanchor="top"),
        margin=dict(t=120),
        legend_title_text="rotor",
    )
    return apply_theme(fig, theme)


def plot_landing_dispersion(
    landings: Any,
    target: tuple[float, float] = (0.0, 0.0),
    *,
    containment: tuple[float, ...] = (0.5, 0.95),
    theme: str | None = None,
    title: str | None = None,
) -> Any:
    """Scatter of landing points with circles that contain given fractions.

    Parameters
    ----------
    landings:
        Array of shape ``(n, 2)`` of horizontal landing positions (m).
    target:
        Intended landing point.
    containment:
        Fractions of the landings each circle (centred on ``target``) should
        contain; the radius is the matching quantile of the miss distance.
    """
    go = _plotly.require_go()
    pts = np.asarray(landings, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("landings must have shape (n, 2).")
    tx, ty = float(target[0]), float(target[1])
    miss = np.hypot(pts[:, 0] - tx, pts[:, 1] - ty)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name="landings",
                   marker=dict(size=7, opacity=0.8))
    )
    fig.add_trace(
        go.Scatter(x=[tx], y=[ty], mode="markers", name="target",
                   marker=dict(symbol="x", size=12))
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 120)
    for frac in containment:
        if not 0.0 < frac < 1.0:
            raise ValueError("containment fractions must be in (0, 1).")
        r = float(np.quantile(miss, frac))
        fig.add_trace(
            go.Scatter(
                x=tx + r * np.cos(angle), y=ty + r * np.sin(angle), mode="lines",
                name=f"{frac:.0%} within {r:.2f} m", line=dict(dash="dot"),
            )
        )
    fig.update_layout(
        title=title or "Landing dispersion",
        xaxis_title="x (m)", yaxis_title="y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return apply_theme(fig, theme)
