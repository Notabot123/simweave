"""simweave: atomic-clock hybrid simulation engine.

Public surface re-exports the primitives most users will need. For anything
else, import the submodule directly (``simweave.continuous``, ``simweave.agents``,
etc).
"""

from simweave.core import (
    Clock,
    EventQueue,
    ScheduledEvent,
    Entity,
    SimEnvironment,
    configure as configure_logging,
    get_logger,
)
from simweave.discrete import (
    EntityProperties,
    exponential,
    uniform,
    normal,
    deterministic,
    set_default_seed,
    Queue,
    PriorityQueue,
    Resource,
    ResourcePool,
    Service,
    ArrivalGenerator,
)
from simweave.continuous import (
    DynamicSystem,
    SupportsDynamics,
    SimulationResult,
    simulate,
    ContinuousProcess,
    MassSpringDamper,
    SimplePendulum,
    QuarterCarModel,
    SeriesRLC,
    ThermalRC,
    TwoMassThermal,
)
from simweave.spatial import Graph, grid_graph
from simweave.agents import (
    Compass,
    Agent,
    a_star,
    dijkstra,
    NoPathError,
    manhattan,
    euclidean,
    chebyshev,
)
from simweave.supplychain import InventoryItems, Warehouse
from simweave.mc import MCResult, run_monte_carlo, run_batched_mc
from simweave.units import (
    SIUnit,
    Distance,
    Velocity,
    Acceleration,
    Mass,
    Force,
    Area,
    Volume,
    TimeUnit,
)
from simweave.currency import (
    Money,
    CurrencyMismatchError,
    FXConverter,
    StaticFXConverter,
    CallableFXConverter,
    format_money,
    register_custom,
    unregister_custom,
    is_valid_currency,
    get_decimals,
    list_codes,
)
# simweave.viz is import-cheap: plotly is only required when a plot helper
# is actually called. The names below are always importable; calling one
# without the `simweave[viz]` extra raises a clear ImportError.
from simweave.viz import (
    QueueLengthRecorder,
    ServiceUtilisationRecorder,
    WarehouseStockRecorder,
    apply_theme,
    available_themes,
    get_default_theme,
    have_plotly,
    plot_agent_path,
    plot_mc_fan,
    plot_phase_portrait,
    plot_queue_length,
    plot_service_utilisation,
    plot_state_trajectories,
    plot_warehouse_stock,
    register_theme,
    set_default_theme,
)

__version__ = "0.5.0"

__all__ = [
    # Core
    "Clock",
    "EventQueue",
    "ScheduledEvent",
    "Entity",
    "SimEnvironment",
    "configure_logging",
    "get_logger",
    # Discrete
    "EntityProperties",
    "exponential",
    "uniform",
    "normal",
    "deterministic",
    "set_default_seed",
    "Queue",
    "PriorityQueue",
    "Resource",
    "ResourcePool",
    "Service",
    "ArrivalGenerator",
    # Continuous
    "DynamicSystem",
    "SupportsDynamics",
    "SimulationResult",
    "simulate",
    "ContinuousProcess",
    "MassSpringDamper",
    "SimplePendulum",
    "QuarterCarModel",
    "SeriesRLC",
    "ThermalRC",
    "TwoMassThermal",
    # Spatial / Agents
    "Graph",
    "grid_graph",
    "Compass",
    "Agent",
    "a_star",
    "dijkstra",
    "NoPathError",
    "manhattan",
    "euclidean",
    "chebyshev",
    # Supply chain
    "InventoryItems",
    "Warehouse",
    # Monte Carlo
    "MCResult",
    "run_monte_carlo",
    "run_batched_mc",
    # Units
    "SIUnit",
    "Distance",
    "Velocity",
    "Acceleration",
    "Mass",
    "Force",
    "Area",
    "Volume",
    "TimeUnit",
    # Currency
    "Money",
    "CurrencyMismatchError",
    "FXConverter",
    "StaticFXConverter",
    "CallableFXConverter",
    "format_money",
    "register_custom",
    "unregister_custom",
    "is_valid_currency",
    "get_decimals",
    "list_codes",
    # Viz
    "QueueLengthRecorder",
    "ServiceUtilisationRecorder",
    "WarehouseStockRecorder",
    "apply_theme",
    "available_themes",
    "get_default_theme",
    "have_plotly",
    "plot_agent_path",
    "plot_mc_fan",
    "plot_phase_portrait",
    "plot_queue_length",
    "plot_service_utilisation",
    "plot_state_trajectories",
    "plot_warehouse_stock",
    "register_theme",
    "set_default_theme",
    # Meta
    "__version__",
]
