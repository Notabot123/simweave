"""``simweave.reliability`` -- maintainability and availability modelling.

Classes
-------
* :class:`SubsystemSpec` -- immutable description of one fitted subsystem.
* :class:`SubsystemState` -- UP / AWAITING_PART / IN_REPAIR enum.
* :class:`SubsystemStatus` -- live runtime state of one subsystem.
* :class:`ReliableEntity` -- Entity subclass with failable subsystems.
* :class:`RepairJob` -- work item that flows through a :class:`RepairCentre`.
* :class:`RepairCentre` -- Service subclass for repair / new-buy operations.
* :class:`Fleet` -- collection of :class:`ReliableEntity` with aggregate metrics.
* :class:`FleetAvailabilityRecorder` -- time-series recorder for the fleet.
* :class:`SweepResult` -- result of a sensitivity sweep.
* :func:`sensitivity_sweep` -- 1-D / 2-D parameter sweep with MC averaging.
"""

from simweave.reliability.subsystem import SubsystemSpec, SubsystemState, SubsystemStatus
from simweave.reliability.repair import RepairJob, RepairCentre
from simweave.reliability.entity import ReliableEntity
from simweave.reliability.fleet import Fleet, FleetAvailabilityRecorder
from simweave.reliability.sensitivity import SweepResult, sensitivity_sweep

__all__ = [
    "SubsystemSpec",
    "SubsystemState",
    "SubsystemStatus",
    "RepairJob",
    "RepairCentre",
    "ReliableEntity",
    "Fleet",
    "FleetAvailabilityRecorder",
    "SweepResult",
    "sensitivity_sweep",
]
