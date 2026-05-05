"""Parameter sensitivity analysis for discrete simulations.

:func:`sensitivity_sweep` varies one or two scalar parameters of a scenario
builder function across a grid, collecting a scalar metric from each cell.
Monte Carlo averaging is supported: pass ``n_runs > 1`` to run multiple
stochastic replicates per grid point and obtain mean ± standard deviation.

The result is a :class:`SweepResult` that can be passed directly to
:func:`~simweave.viz.plots.plot_sensitivity_surface` or
:func:`~simweave.viz.plots.plot_sensitivity_heatmap` for visualisation.

Example (1-D sweep)::

    from simweave.reliability.sensitivity import sensitivity_sweep

    def build(fleet_size, seed):
        # ... build and run scenario, return Ao ...
        return operational_availability

    result = sensitivity_sweep(
        build,
        param1_name="fleet_size",
        param1_values=[5, 10, 15, 20, 25],
        metric_name="Ao",
        n_runs=20,
    )

Example (2-D sweep)::

    def build(repair_bays, stock_level, seed):
        # ...
        return operational_availability

    result = sensitivity_sweep(
        build,
        param1_name="repair_bays",
        param1_values=[1, 2, 3, 4],
        param2_name="stock_level",
        param2_values=[2, 4, 6, 8, 10],
        metric_name="Ao",
        n_runs=30,
    )
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class SweepResult:
    """Result of a 1-D or 2-D sensitivity sweep.

    Attributes
    ----------
    param1_name:
        Name of the first swept parameter.
    param1_values:
        Array of values swept for parameter 1.
    param2_name:
        Name of the second parameter, or ``None`` for a 1-D sweep.
    param2_values:
        Array of values swept for parameter 2, or ``None`` for a 1-D sweep.
    metric_name:
        Label for the output metric (used in plot axis titles).
    metric_mean:
        Mean metric value.  Shape ``(n1,)`` for 1-D or ``(n1, n2)`` for 2-D.
    metric_std:
        Standard deviation across MC replicates.  All zeros when
        ``n_runs == 1``.  Same shape as ``metric_mean``.
    n_runs:
        Number of Monte Carlo replicates per grid point.
    """

    param1_name: str
    param1_values: np.ndarray
    param2_name: str | None
    param2_values: np.ndarray | None
    metric_name: str
    metric_mean: np.ndarray
    metric_std: np.ndarray
    n_runs: int = 1

    @property
    def is_2d(self) -> bool:
        return self.param2_name is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_task(payload: tuple[Callable, Any, Any, int]) -> float:
    fn, v1, v2, seed = payload
    if v2 is not None:
        return float(fn(v1, v2, seed))
    return float(fn(v1, seed))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sensitivity_sweep(
    scenario_builder: Callable[..., float],
    param1_name: str,
    param1_values: Sequence[float],
    param2_name: str | None = None,
    param2_values: Sequence[float] | None = None,
    metric_name: str = "metric",
    n_runs: int = 1,
    seed: int = 0,
    executor: str = "serial",
    n_workers: int | None = None,
) -> SweepResult:
    """Run a 1-D or 2-D parameter sensitivity sweep with optional MC averaging.

    Parameters
    ----------
    scenario_builder:
        Callable with signature ``f(p1, seed) -> float`` for a 1-D sweep,
        or ``f(p1, p2, seed) -> float`` for a 2-D sweep.  Must return a
        scalar metric (e.g. operational availability).  Must be picklable
        when ``executor="processes"``.
    param1_name:
        Name of the first parameter (used in plot labels).
    param1_values:
        Values to sweep for parameter 1.
    param2_name:
        Name of the second parameter.  ``None`` → 1-D sweep.
    param2_values:
        Values to sweep for parameter 2.  Required when ``param2_name``
        is not ``None``.
    metric_name:
        Label for the output metric.
    n_runs:
        Number of Monte Carlo replicates per grid point.  Each replicate
        receives a unique seed derived from the base ``seed``.
    seed:
        Base random seed.  Replicate *r* at grid point *i* (or *(i, j)*)
        receives seed ``seed + r + i * n_runs`` (or similar) to ensure
        independence across the grid.
    executor:
        ``"serial"`` (default) or ``"processes"`` for multi-core parallelism.
    n_workers:
        Number of worker processes.  ``None`` → OS default.

    Returns
    -------
    SweepResult
    """
    p1 = np.asarray(param1_values, dtype=float)
    is_2d = param2_name is not None and param2_values is not None
    p2 = np.asarray(param2_values, dtype=float) if is_2d else None

    if param2_name is not None and param2_values is None:
        raise ValueError("param2_values is required when param2_name is given.")

    # Build flat task list: (builder, v1, v2_or_None, seed_for_replicate)
    tasks: list[tuple] = []
    if is_2d:
        assert p2 is not None
        for i, v1 in enumerate(p1):
            for j, v2 in enumerate(p2):
                for r in range(n_runs):
                    tasks.append(
                        (scenario_builder, v1, v2, seed + i * len(p2) * n_runs + j * n_runs + r)
                    )
    else:
        for i, v1 in enumerate(p1):
            for r in range(n_runs):
                tasks.append((scenario_builder, v1, None, seed + i * n_runs + r))

    # Execute
    if executor == "processes" and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            raw = list(ex.map(_run_task, tasks))
    else:
        raw = [_run_task(t) for t in tasks]

    raw_arr = np.asarray(raw, dtype=float)

    if is_2d:
        assert p2 is not None
        raw_arr = raw_arr.reshape(len(p1), len(p2), n_runs)
        mean = raw_arr.mean(axis=2)
        std = raw_arr.std(axis=2)
    else:
        raw_arr = raw_arr.reshape(len(p1), n_runs)
        mean = raw_arr.mean(axis=1)
        std = raw_arr.std(axis=1)

    return SweepResult(
        param1_name=param1_name,
        param1_values=p1,
        param2_name=param2_name,
        param2_values=p2,
        metric_name=metric_name,
        metric_mean=mean,
        metric_std=std,
        n_runs=n_runs,
    )


__all__ = ["SweepResult", "sensitivity_sweep"]
