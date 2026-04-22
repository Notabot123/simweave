"""Tests for the steady-state reorder-point / cost-optimisation helpers.

These tests are skipped when scipy isn't installed, matching the library's
stance that scipy is an optional extra (``simeng[optim]``).
"""
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

# The imports below are intentionally placed after the scipy skip-gate so that
# the whole file is bypassed cleanly when scipy isn't installed.
from simeng.supplychain.inventory import InventoryItems  # noqa: E402
from simeng.supplychain.warehouse import Warehouse  # noqa: E402
from simeng.supplychain.optimization import (  # noqa: E402
    poisson_reorder_points,
    cost_optimise_stock,
    pareto_sweep,
    cost_optimise_stock_sim,
)


def _warehouse_with_demand(n=3, rate=1.0):
    inv = InventoryItems(
        part_names=[f"sku_{i}" for i in range(n)],
        unit_cost=[1.0 + i for i in range(n)],
        stock_level=[10.0] * n,
        batchsize=[5.0] * n,
        reorder_points=[2.0] * n,
        repairable_prc=[0.0] * n,
        repair_times=[0.0] * n,
        newbuy_leadtimes=[3.0] * n,
    )
    w = Warehouse(inv, name="w")
    w._demand_rate = np.full(n, rate)
    return w


def test_poisson_returns_monotone_in_availability():
    w = _warehouse_with_demand()
    k_lo, _ = poisson_reorder_points(w, target_availability=0.5)
    k_hi, _ = poisson_reorder_points(w, target_availability=0.95)
    # Higher availability target => at least as much safety stock.
    assert np.all(k_hi >= k_lo)


def test_poisson_rejects_bad_target():
    w = _warehouse_with_demand()
    with pytest.raises(ValueError):
        poisson_reorder_points(w, target_availability=1.5)


def test_poisson_requires_demand_rate():
    inv = InventoryItems(
        part_names=["a"], unit_cost=[1.0], stock_level=[10.0],
        batchsize=[5.0], reorder_points=[2.0], repairable_prc=[0.0],
        repair_times=[0.0], newbuy_leadtimes=[1.0],
    )
    w = Warehouse(inv)
    with pytest.raises(ValueError):
        poisson_reorder_points(w, target_availability=0.9)


def test_poisson_assign_mutates_inventory():
    w = _warehouse_with_demand(n=2, rate=1.0)
    original_rop = w.inv.reorder_points.copy()
    k, _ = poisson_reorder_points(w, target_availability=0.9, assign=True)
    assert np.array_equal(w.inv.reorder_points, k)
    # Stock set to k + batchsize.
    assert np.array_equal(w.inv.stock_level, k + w.inv.batchsize)
    # Confirm we actually changed it.
    assert not np.array_equal(w.inv.reorder_points, original_rop) or np.all(k == original_rop)


def test_cost_optimise_stock_runs():
    w = _warehouse_with_demand(n=2, rate=1.0)
    solution, cost = cost_optimise_stock(
        w, target_availability=0.8, maxiter=20, seed=0,
    )
    assert solution.shape == (2,)
    assert np.all(solution >= 0)
    assert cost > 0


def test_cost_optimise_stock_sim_respects_bounds():
    # Minimise a trivial quadratic so we know the answer.
    def sim_cost(x):
        target = np.array([2.0, 4.0])
        return float(np.sum((x - target) ** 2))

    x, c = cost_optimise_stock_sim(
        sim_cost, lower=np.zeros(2), upper=np.full(2, 10.0),
        maxiter=30, seed=0,
    )
    assert x.shape == (2,)
    assert c < 1.0  # should get close to the target


def test_pareto_sweep_returns_expected_keys():
    w = _warehouse_with_demand(n=2, rate=1.0)
    out = pareto_sweep(w, availability_range=np.array([0.5, 0.8]))
    assert set(out) == {"availability", "cost_cost_optimal", "cost_poisson"}
    assert out["availability"].shape == (2,)
    assert out["cost_cost_optimal"].shape == (2,)
