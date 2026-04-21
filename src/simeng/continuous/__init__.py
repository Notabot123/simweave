"""Continuous-time dynamics: state-space solver, plug-in systems, hybrid wrapper."""
from simeng.continuous.solver import (
    DynamicSystem,
    SupportsDynamics,
    SimulationResult,
    simulate,
    ContinuousProcess,
)
from simeng.continuous.systems import (
    MassSpringDamper,
    SimplePendulum,
    QuarterCarModel,
    SeriesRLC,
    ThermalRC,
    TwoMassThermal,
)

__all__ = [
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
]
