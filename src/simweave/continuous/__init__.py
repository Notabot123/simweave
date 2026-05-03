"""Continuous-time dynamics: state-space solver, plug-in systems, hybrid wrapper."""

from simweave.continuous.solver import (
    DynamicSystem,
    SupportsDynamics,
    SimulationResult,
    simulate,
    ContinuousProcess,
    PIDController,
)
from simweave.continuous.systems import (
    MassSpringDamper,
    SimplePendulum,
    QuarterCarModel,
    HalfCarModel,
    RollCarModel,
    FullCarModel,
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
    "PIDController",
    "MassSpringDamper",
    "SimplePendulum",
    "QuarterCarModel",
    "HalfCarModel",
    "RollCarModel",
    "FullCarModel",
    "SeriesRLC",
    "ThermalRC",
    "TwoMassThermal",
]
