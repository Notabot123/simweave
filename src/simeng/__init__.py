"""simeng: atomic-clock hybrid simulation engine.

Public surface re-exports the primitives most users will need. For anything
else, import the submodule directly (``simeng.continuous``, ``simeng.agents``,
etc).
"""

from simeng.core import (
    Clock,
    EventQueue,
    ScheduledEvent,
    Entity,
    SimEnvironment,
    configure as configure_logging,
    get_logger,
)
from simeng.discrete import (
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
from simeng.continuous import (
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
from simeng.spatial import Graph, grid_graph
from simeng.agents import (
    Compass,
    Agent,
    a_star,
    dijkstra,
    NoPathError,
    manhattan,
    euclidean,
    chebyshev,
)
from simeng.supplychain import InventoryItems, Warehouse
from simeng.mc import MCResult, run_monte_carlo, run_batched_mc
from simeng.units import (
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
from simeng.currency import (
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

__version__ = "0.1.0"

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
    # Meta
    "__version__",
]
