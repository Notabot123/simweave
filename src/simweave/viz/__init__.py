"""``simweave.viz`` -- plotly-based visualisation helpers.

Every helper returns a ``plotly.graph_objects.Figure`` so JS frontends
(such as EdgeWeave) can consume the figure via ``fig.to_json()`` without
losing structure. Themes are applied via a small registry so users can
switch between light, dark and custom palettes without rebuilding plots.

Importing this module is cheap; plotly is only required when a plot
helper is actually called. Install via the optional extra::

    pip install simweave[viz]

Quick start::

    from simweave.viz import (
        plot_state_trajectories,
        set_default_theme,
    )
    from simweave.continuous import simulate, MassSpringDamper

    res = simulate(MassSpringDamper(), t_span=(0, 10), dt=0.01)
    set_default_theme("dark")
    fig = plot_state_trajectories(res)
    fig.show()              # or fig.write_html("msd.html"), or fig.to_json()
"""

from __future__ import annotations

from simweave.viz.themes import (
    Theme,
    apply_theme,
    available_themes,
    get_default_theme,
    get_theme,
    register_theme,
    set_default_theme,
)
from simweave.viz.recorders import (
    QueueLengthRecorder,
    ServiceUtilisationRecorder,
    WarehouseStockRecorder,
)
from simweave.viz.plots import (
    plot_agent_path,
    plot_mc_fan,
    plot_phase_portrait,
    plot_queue_length,
    plot_service_utilisation,
    plot_state_trajectories,
    plot_warehouse_stock,
)
from simweave.viz._plotly import have_plotly

__all__ = [
    # Themes
    "Theme",
    "apply_theme",
    "available_themes",
    "get_default_theme",
    "get_theme",
    "register_theme",
    "set_default_theme",
    # Recorders
    "QueueLengthRecorder",
    "ServiceUtilisationRecorder",
    "WarehouseStockRecorder",
    # Plot helpers
    "plot_agent_path",
    "plot_mc_fan",
    "plot_phase_portrait",
    "plot_queue_length",
    "plot_service_utilisation",
    "plot_state_trajectories",
    "plot_warehouse_stock",
    # Probe
    "have_plotly",
]
