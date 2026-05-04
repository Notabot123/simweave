from __future__ import annotations

import numpy as np

from simweave.viz import _plotly
from simweave.viz.themes import apply_theme
from simweave.analysis.vehicle import compute_vehicle_metrics

from simweave.units.si import (
        Acceleration,
        Angle,
        Distance,
        Force,
    )

from plotly.subplots import make_subplots


def plot_vehicle_metrics(
    result,
    *,
    model=None,
    theme: str | None = None,
    title: str | None = None,
):
    """Plot vehicle metrics with adaptive layout and automatic units."""

    go = _plotly.require_go()

    metrics = compute_vehicle_metrics(result, model)
    t = np.asarray(result.time)

    # --- unit helpers ---
    def accel(x): return Acceleration(x).to("m/s^2")
    def angle(x): return Angle(x).to("deg")
    def dist(x): return Distance(x).to("m")
    def force(x): return Force(x).to("N")

    # --- detect signals ---
    has_pitch = metrics["pitch"] is not None
    has_roll = metrics["roll"] is not None

    # --- build subplot structure ---
    subplot_titles = ["Body acceleration (comfort)"]

    if has_pitch or has_roll:
        subplot_titles.append("Pitch / Roll")

    subplot_titles.append("Suspension travel")
    subplot_titles.append(metrics["tyre_metric"]["name"])

    n_rows = len(subplot_titles)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=tuple(subplot_titles),
    )

    row = 1

    # --- 1. BODY ACCEL ---
    fig.add_trace(
        go.Scatter(x=t, y=accel(metrics["body_accel"]), mode="lines",
                   name="body accel [m/s²]"),
        row=row, col=1,
    )

    rms = accel(metrics["body_accel_RMS"])
    fig.add_annotation(
        text=f"RMS: {rms:.3f} m/s²",
        xref="paper", yref="paper",
        x=0.01, y=0.98,
        showarrow=False,
    )

    fig.update_yaxes(title_text="m/s²", row=row)
    row += 1

    # --- 2. PITCH / ROLL (optional) ---
    if has_pitch or has_roll:
        if has_pitch:
            fig.add_trace(
                go.Scatter(x=t, y=angle(metrics["pitch"]),
                           mode="lines", name="pitch [deg]"),
                row=row, col=1,
            )

        if has_roll:
            fig.add_trace(
                go.Scatter(x=t, y=angle(metrics["roll"]),
                           mode="lines", name="roll [deg]"),
                row=row, col=1,
            )

        fig.update_yaxes(title_text="deg", row=row)
        row += 1

    # --- 3. SUSPENSION TRAVEL ---
    for key, data in metrics["suspension_travel"].items():
        fig.add_trace(
            go.Scatter(
                x=t,
                y=dist(data),
                mode="lines",
                name=f"{key.upper()} travel [m]",
            ),
            row=row, col=1,
        )

    fig.update_yaxes(title_text="m", row=row)
    row += 1

    # --- 4. TYRE ---
    tyre_metric = metrics["tyre_metric"]

    if tyre_metric["unit"] == "N":
        conv = force
        unit_label = "N"
    else:
        conv = dist
        unit_label = "m"

    for key, data in metrics["tyre"].items():
        fig.add_trace(
            go.Scatter(
                x=t,
                y=conv(data),
                mode="lines",
                line={"dash": "dot"},
                name=f"{key.upper()} tyre [{unit_label}]",
            ),
            row=row, col=1,
        )

    fig.update_yaxes(title_text=unit_label, row=row)

    # --- layout ---
    fig.update_layout(
        title=title or "Vehicle metrics",
        xaxis_title="time",
        legend_title_text="signals",
        height=300 * n_rows,  # dynamic height
    )

    return apply_theme(fig, theme)