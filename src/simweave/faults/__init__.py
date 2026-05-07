"""Fault injection for predictive-maintenance dataset generation.

Inject physically-meaningful parameter faults into continuous dynamic systems
to produce labelled time-series suitable for training ML models (binary
classifier, RUL regression, LSTM, etc.).

Typical usage
-------------
1. Describe the degradation with :class:`FaultProfile`.
2. Map it to a system parameter with :class:`ParameterFault`.
3. Wrap the system in a :class:`FaultInjector` and run
   :func:`~simweave.continuous.solver.simulate`.
4. Assemble the labelled :class:`FaultDataset` and export to a DataFrame.
"""

from simweave.faults.fault import FaultProfile, ParameterFault
from simweave.faults.injector import FaultInjector
from simweave.faults.recorder import FaultRecorder
from simweave.faults.dataset import FaultDataset

__all__ = [
    "FaultProfile",
    "ParameterFault",
    "FaultInjector",
    "FaultRecorder",
    "FaultDataset",
]
