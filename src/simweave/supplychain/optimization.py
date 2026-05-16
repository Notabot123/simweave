"""Steady-state reorder-point heuristics and cost-optimal sweeps.

Two complementary approaches:

* :func:`poisson_reorder_points` -- a closed-form Poisson heuristic that
  treats each SKU independently. Fast, decent when demands are independent.
* :func:`cost_optimise_stock` -- a differential-evolution search over the
  stock vector, subject to a global availability constraint. Respects the
  fact that cheap items should cover much of your availability risk so
  expensive ones can be sparse. Requires scipy.

* :func:`pareto_sweep` -- vary the availability target across a range and
  plot the cost-availability trade-off.

These are *approximations* for planning purposes. For true sim-based
optimisation, wrap your simulation in ``simulate_cost(stock_levels)`` and
pass that to :func:`cost_optimise_stock_sim`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simweave.supplychain.warehouse import Warehouse


def _ensure_scipy():
    try:
        import scipy.stats as stats  # noqa: F401
        from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "simweave.supplychain.optimization requires scipy. "
            "Install with `pip install simweave[optim]` or `pip install scipy`."
        ) from e


def poisson_reorder_points(
    warehouse: Warehouse,
    target_availability: float = 0.9,
    quantize_by_batchsize: bool = False,
    assign: bool = False,
) -> tuple[np.ndarray, float]:
    """Return the Poisson-based reorder points ``k`` for each SKU.

    The per-SKU availability target ``p = target_availability ** (1/n_items)``
    ensures the joint probability of all SKUs being in stock meets the
    overall target when demands are independent.
    """
    _ensure_scipy()
    import scipy.stats as stats

    if not 0.0 < target_availability < 0.999:
        raise ValueError("target_availability must lie in (0, 0.999).")
    if warehouse._demand_rate is None or float(np.sum(warehouse._demand_rate)) == 0.0:
        raise ValueError(
            "Demand rate unknown. Run a simulation first, then call "
            "warehouse.estimate_demand_rate(total_sim_time)."
        )

    demand = warehouse._demand_rate
    inv = warehouse.inv
    logistics_delay = (
        inv.repairable_prc * inv.repair_times
        + (1 - inv.repairable_prc) * inv.newbuy_leadtimes
    )
    lam = demand * logistics_delay
    per_item_avail = target_availability ** (1.0 / inv.n_items)

    k = stats.poisson.ppf(per_item_avail, lam)
    if quantize_by_batchsize:
        k = np.floor(k / inv.batchsize) * inv.batchsize

    total_cost = float(np.sum(inv.unit_cost * (inv.batchsize + k)))

    if assign:
        inv.reorder_points = k
        inv.stock_level = k + inv.batchsize

    return k, total_cost

def _objective(x: np.ndarray, inv, quant) -> float:
    return float(np.sum(inv.unit_cost * np.round(x) * quant))

def _availability_con(x: np.ndarray, lam) -> float:
    import scipy.stats as stats
    return float(np.prod(stats.poisson.cdf(np.round(x), lam)))

def cost_optimise_stock(
    warehouse: Warehouse,
    target_availability: float = 0.9,
    quantize_by_batchsize: bool = False,
    assign: bool = False,
    ub: float = 100.0,
    seed: int | None = 1,
    maxiter: int = 200,
    workers: int = -1,
) -> tuple[np.ndarray, float]:
    """Differential-evolution cost minimisation with availability constraint.

    This balances cost vs availability on a whole-warehouse basis rather than
    per item -- cheap items should cover more risk, expensive items less.

    workers defaults to -1 which is using all available cores (fastest).
    For reproducibility, use 1.
    Specify workers as desired on shared devices.
    """
    _ensure_scipy()
    from functools import partial
    from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds

    if not 0.0 < target_availability < 0.999:
        raise ValueError("target_availability must lie in (0, 0.999).")
    if warehouse._demand_rate is None or float(np.sum(warehouse._demand_rate)) == 0.0:
        raise ValueError(
            "Demand rate unknown. Run a simulation first, then call "
            "warehouse.estimate_demand_rate(total_sim_time)."
        )

    inv = warehouse.inv
    demand = warehouse._demand_rate
    logistics_delay = (
        inv.repairable_prc * inv.repair_times
        + (1 - inv.repairable_prc) * inv.newbuy_leadtimes
    )
    lam = demand * logistics_delay
    quant = inv.batchsize if quantize_by_batchsize else np.ones(inv.n_items)

    ## v0.8.2 we moved _objective and _availability_con outside this function, to be pickable, required for workers>1
    ## partial is required, as scipy multiprocessing won't pickle lambda either (or any local-defined callable)
    obj = partial(_objective, inv=inv, quant=quant)
    con = partial(_availability_con, lam=lam)

    bounds = Bounds(lb=np.zeros(inv.n_items), ub=ub * np.ones(inv.n_items))
    nlc = NonlinearConstraint(con, target_availability, 1.0)

    result = differential_evolution(
        obj,
        bounds,
        constraints=nlc,
        seed=seed,
        maxiter=maxiter,
        polish=False,
        updating="deferred",
        workers=workers
    )


    solution = np.round(result.x * quant)
    total_cost = float(np.sum(inv.unit_cost * (inv.batchsize + solution)))

    if assign:
        inv.reorder_points = solution
        inv.stock_level = solution + inv.batchsize

    return solution, total_cost


def pareto_sweep(
    warehouse: Warehouse,
    availability_range: np.ndarray | None = None,
    *,
    method: str = "both",
) -> dict[str, np.ndarray]:
    """Sweep availability targets and return costs for Poisson and/or DE."""

    if availability_range is None:
        availability_range = np.clip(np.arange(0.1, 1.0, 0.05), 0.0, 0.99)

    do_de = method in ("both", "de")
    do_poisson = method in ("both", "poisson")

    costs_de = []
    costs_poisson = []

    for a in availability_range:
        if do_de:
            _, c_de = cost_optimise_stock(warehouse, target_availability=float(a))
            costs_de.append(c_de)
        if do_poisson:
            _, c_p = poisson_reorder_points(warehouse, target_availability=float(a))
            costs_poisson.append(c_p)

    out = {"availability": np.asarray(availability_range)}

    if do_de:
        out["cost_cost_optimal"] = np.asarray(costs_de)
    if do_poisson:
        out["cost_poisson"] = np.asarray(costs_poisson)

    return out


# ---------------------------------------------------------------------------
# Simulation-based optimisation hook.
# ---------------------------------------------------------------------------


def cost_optimise_stock_sim(
    simulate_cost: Callable[[np.ndarray], float],
    lower: np.ndarray,
    upper: np.ndarray,
    maxiter: int = 50,
    seed: int | None = 1,
) -> tuple[np.ndarray, float]:
    """Optimise stock levels using a user-supplied simulation callback.

    Parameters
    ----------
    simulate_cost:
        A callable accepting a 1-D ndarray of stock levels and returning the
        cost (lower is better, availability implicit in the simulated
        behaviour).
    lower, upper:
        Per-SKU bounds for the optimiser.
    """
    _ensure_scipy()
    from scipy.optimize import differential_evolution, Bounds

    bounds = Bounds(
        lb=np.asarray(lower, dtype=float), ub=np.asarray(upper, dtype=float)
    )
    result = differential_evolution(
        simulate_cost, bounds, seed=seed, maxiter=maxiter, polish=False
    )
    return result.x, float(result.fun)
