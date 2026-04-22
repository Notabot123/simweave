"""Monte Carlo harness.

Three execution strategies are supported, trading simplicity for throughput:

* ``"serial"`` -- one process, one thread. Useful for tiny sweeps and for
  reproducibility when debugging.
* ``"processes"`` -- ``concurrent.futures.ProcessPoolExecutor``. Each replicate
  runs in its own OS process so the GIL doesn't bind us. The per-task
  pickling overhead is negligible once the inner sim runs for longer than a
  second or so.
* ``"threads"`` -- ``concurrent.futures.ThreadPoolExecutor``. Only useful for
  I/O-bound scenarios (e.g. hitting a remote data store before each
  replicate). Pure-Python compute is GIL-bound and will not scale this way.
  On Python 3.13's free-threading build the GIL restriction goes away, so
  this becomes a legitimate option for compute too.

Seeding: each replicate receives its own integer seed, used to build a fresh
``numpy.random.Generator``. Pass the seed into your ``scenario_builder`` and
use it to instantiate the generator so results are deterministic per seed.

For the fastest Monte Carlo, rather than using this runner at all, model your
state as arrays with a ``replicate`` axis and step all N replicates in a
single numpy operation. That typically beats process-pool parallelism by
10-100x. See the README for guidance.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np


@dataclass
class MCResult:
    """Monte Carlo summary."""

    n_runs: int
    seeds: list[int]
    samples: np.ndarray | list[Any]
    scenario_name: str = "default"
    extras: dict[str, Any] = field(default_factory=dict)

    def mean(self, axis: int = 0) -> np.ndarray:
        arr = np.asarray(self.samples)
        return arr.mean(axis=axis)

    def std(self, axis: int = 0) -> np.ndarray:
        arr = np.asarray(self.samples)
        return arr.std(axis=axis)

    def quantile(self, q: float | list[float], axis: int = 0) -> np.ndarray:
        arr = np.asarray(self.samples)
        return np.quantile(arr, q, axis=axis)


def _run_single(payload: tuple[Callable[[int], Any], int]) -> Any:
    builder, seed = payload
    return builder(seed)


def run_monte_carlo(scenario_builder: Callable[[int], Any],
                     n_runs: int,
                     seeds: Iterable[int] | None = None,
                     executor: str = "serial",
                     n_workers: int | None = None,
                     scenario_name: str = "default") -> MCResult:
    """Run Monte Carlo replicates.

    Parameters
    ----------
    scenario_builder:
        Callable ``f(seed) -> Any``. Must be picklable when ``executor="processes"``.
    n_runs:
        Number of replicates.
    seeds:
        Optional iterable of seeds. Defaults to ``range(n_runs)``.
    executor:
        One of ``"serial"``, ``"processes"``, ``"threads"``.
    n_workers:
        Max workers for pool executors. Defaults to os-determined.
    scenario_name:
        Label carried on the result for bookkeeping.
    """
    if executor not in {"serial", "processes", "threads"}:
        raise ValueError("executor must be one of 'serial' | 'processes' | 'threads'.")
    seed_list = list(seeds) if seeds is not None else list(range(n_runs))
    if len(seed_list) != n_runs:
        raise ValueError("len(seeds) must equal n_runs.")

    payload = [(scenario_builder, s) for s in seed_list]

    if executor == "serial" or n_runs == 1:
        results = [_run_single(p) for p in payload]
    elif executor == "processes":
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_run_single, payload))
    else:  # threads
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_run_single, payload))

    try:
        samples: np.ndarray | list[Any] = np.asarray(results)
        # If numpy fell back to object dtype, keep a Python list for clarity.
        if samples.dtype == object:
            samples = list(results)
    except Exception:
        samples = list(results)

    return MCResult(
        n_runs=n_runs,
        seeds=seed_list,
        samples=samples,
        scenario_name=scenario_name,
    )


# ---------------------------------------------------------------------------
# Batched / vectorised Monte Carlo helper.
# ---------------------------------------------------------------------------

def run_batched_mc(batched_step: Callable[[np.random.Generator, int], np.ndarray],
                    n_runs: int,
                    seed: int | None = 0,
                    scenario_name: str = "batched") -> MCResult:
    """Run a vectorised Monte Carlo where ``batched_step`` returns an
    ``(n_runs, ...)`` ndarray, all replicates progressed in one numpy op.

    This is a thin wrapper -- the point is to give a single entry point with
    an ``MCResult`` back so callers don't have to distinguish between the two
    styles downstream.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(batched_step(rng, n_runs))
    if samples.shape[0] != n_runs:
        raise ValueError(
            f"batched_step returned first dim {samples.shape[0]}, expected {n_runs}."
        )
    return MCResult(
        n_runs=n_runs,
        seeds=[seed if seed is not None else 0],
        samples=samples,
        scenario_name=scenario_name,
    )
