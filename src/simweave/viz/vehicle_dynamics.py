from __future__ import annotations

import numpy as np

from simweave.viz import _plotly
from simweave.viz.themes import apply_theme

from plotly.subplots import make_subplots


def plot_vehicle_metrics(
    result,
    *,
    model=None,
    theme: str | None = None,
    title: str | None = None,
):
    """Plot vehicle metrics with automatic unit handling."""

    go = _plotly.require_go()
    from simweave.analysis.vehicle import compute_full_car_metrics
    from simweave.units.si import (
        Acceleration,
        Angle,
        Distance,
        Force,
    )

    metrics = compute_full_car_metrics(result, model)
    t = np.asarray(result.time)

    # --- auto unit conversion helpers ---
    def accel(x): return Acceleration(x).to("m/s^2")
    def angle(x): return Angle(x).to("deg")
    def dist(x): return Distance(x).to("m")
    def force(x): return Force(x).to("N")

    # --- create subplots ---
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Body acceleration (comfort)",
            "Pitch / Roll",
            "Suspension travel",
            metrics["tyre_metric"]["name"],
        ),
    )

    # --- 1. BODY ACCEL ---
    y = accel(metrics["body_accel"])
    fig.add_trace(
        go.Scatter(x=t, y=y, mode="lines", name="body accel [m/s²]"),
        row=1, col=1,
    )

    rms = accel(metrics["body_accel_RMS"])
    fig.add_annotation(
        text=f"RMS: {rms:.3f} m/s²",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.98,
        showarrow=False,
    )

    # --- 2. PITCH / ROLL ---
    fig.add_trace(
        go.Scatter(x=t, y=angle(metrics["pitch"]), mode="lines", name="pitch [deg]"),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(x=t, y=angle(metrics["roll"]), mode="lines", name="roll [deg]"),
        row=2, col=1,
    )

    # --- 3. SUSPENSION TRAVEL ---
    for key, data in metrics["suspension_travel"].items():
        fig.add_trace(
            go.Scatter(
                x=t,
                y=dist(data),
                mode="lines",
                name=f"{key.upper()} travel [m]",
            ),
            row=3, col=1,
        )

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
            row=4, col=1,
        )

    # --- layout ---
    fig.update_layout(
        title=title or "Vehicle metrics",
        xaxis_title="time",
        legend_title_text="signals",
        height=900,
    )

    fig.update_yaxes(title_text="m/s²", row=1)
    fig.update_yaxes(title_text="deg", row=2)
    fig.update_yaxes(title_text="m", row=3)
    fig.update_yaxes(title_text=unit_label, row=4)

    return apply_theme(fig, theme)