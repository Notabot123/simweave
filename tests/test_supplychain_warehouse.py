import numpy as np
import pytest

from simeng.core.environment import SimEnvironment
from simeng.supplychain.inventory import InventoryItems
from simeng.supplychain.warehouse import Warehouse


def _inv(n=2, stock=10.0, rop=2.0, batch=5.0, lt=3.0):
    return InventoryItems(
        part_names=[f"sku_{i}" for i in range(n)],
        unit_cost=[1.0] * n,
        stock_level=[stock] * n,
        batchsize=[batch] * n,
        reorder_points=[rop] * n,
        repairable_prc=[0.0] * n,
        repair_times=[0.0] * n,
        newbuy_leadtimes=[lt] * n,
    )


# ---------------------------------------------------------------------------
# InventoryItems
# ---------------------------------------------------------------------------

def test_inventory_coerces_to_ndarray():
    inv = _inv(n=3)
    assert isinstance(inv.stock_level, np.ndarray)
    assert inv.stock_level.dtype == float
    assert inv.n_items == 3


def test_inventory_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        InventoryItems(
            part_names=["a", "b"],
            unit_cost=[1.0],  # too short
            stock_level=[10.0, 10.0],
            batchsize=[5.0, 5.0],
            reorder_points=[2.0, 2.0],
            repairable_prc=[0.0, 0.0],
            repair_times=[0.0, 0.0],
            newbuy_leadtimes=[3.0, 3.0],
        )


def test_inventory_defaults_shelf_life_and_failure_rate():
    inv = _inv(n=2)
    assert np.all(np.isinf(inv.shelf_life))
    assert np.all(inv.failure_rate == 0.0)


# ---------------------------------------------------------------------------
# Warehouse stock ops
# ---------------------------------------------------------------------------

def test_decrement_and_increment():
    w = Warehouse(_inv(n=2, stock=5.0))
    assert w.decrement_by_idx(0, 2.0) is True
    assert w.inv.stock_level[0] == 3.0
    w.increment_by_idx(0, 1.0)
    assert w.inv.stock_level[0] == 4.0


def test_decrement_rejects_when_unavailable():
    w = Warehouse(_inv(n=1, stock=1.0))
    assert w.decrement_by_idx(0, 2.0) is False
    assert w.inv.stock_level[0] == 1.0


def test_decrement_vector_returns_mask():
    w = Warehouse(_inv(n=3, stock=5.0))
    mask = w.decrement_vector(np.array([3.0, 10.0, 1.0]))
    assert mask.tolist() == [True, False, True]
    assert w.inv.stock_level.tolist() == [2.0, 5.0, 4.0]


# ---------------------------------------------------------------------------
# Reorder logic
# ---------------------------------------------------------------------------

def test_reorder_places_order_below_rop():
    w = Warehouse(_inv(n=1, stock=2.0, rop=2.0, batch=5.0, lt=3.0))
    w.process_orders(elapsed=1.0)
    assert w._orders_placed_bool[0]
    assert w._reorders_volume[0] == 5.0


def test_reorder_does_not_duplicate_in_flight():
    w = Warehouse(_inv(n=1, stock=2.0, rop=2.0, batch=5.0, lt=3.0))
    w.process_orders(elapsed=1.0)
    vol_after_first = w._reorders_volume[0]
    w.process_orders(elapsed=1.0)
    assert w._reorders_volume[0] == vol_after_first


def test_reorder_arrives_after_leadtime():
    w = Warehouse(_inv(n=1, stock=2.0, rop=2.0, batch=5.0, lt=3.0))
    w.process_orders(elapsed=1.0)  # place
    w.process_orders(elapsed=1.0)
    w.process_orders(elapsed=1.0)
    w.process_orders(elapsed=1.0)  # delivered
    assert w.inv.stock_level[0] == 2.0 + 5.0
    assert not w._orders_placed_bool[0]


def test_parent_child_backorder_when_parent_empty():
    parent = Warehouse(_inv(n=1, stock=0.0, rop=0.0, batch=5.0, lt=1.0),
                       name="parent")
    child = Warehouse(_inv(n=1, stock=1.0, rop=2.0, batch=5.0, lt=1.0),
                     name="child", parent_warehouse=parent)
    child.process_orders(elapsed=1.0)
    # Parent empty -> no decrement and backorder recorded.
    assert parent.inv.stock_level[0] == 0.0
    assert child._lifetime_backorders[0] == 5.0


def test_parent_child_successful_transfer():
    parent = Warehouse(_inv(n=1, stock=20.0, rop=0.0, batch=5.0, lt=1.0),
                       name="parent")
    child = Warehouse(_inv(n=1, stock=1.0, rop=2.0, batch=5.0, lt=1.0),
                     name="child", parent_warehouse=parent)
    child.process_orders(elapsed=1.0)
    assert parent.inv.stock_level[0] == 15.0
    assert child._reorders_volume[0] == 5.0


def test_estimate_demand_rate():
    w = Warehouse(_inv(n=1, stock=10.0, batch=5.0))
    # Place one order -> lifetime_total_orders grows by batchsize.
    w.inv.stock_level[0] = 1.0
    w.process_orders(elapsed=1.0)
    rate = w.estimate_demand_rate(total_sim_time=10.0)
    # lifetime includes initial stock (10) + reorder batch (5) = 15, divided by 10.
    assert rate[0] == pytest.approx(1.5)


def test_estimate_demand_rate_rejects_zero():
    w = Warehouse(_inv())
    with pytest.raises(ValueError):
        w.estimate_demand_rate(0.0)


# ---------------------------------------------------------------------------
# Integration with SimEnvironment
# ---------------------------------------------------------------------------

def test_warehouse_in_environment_ticks():
    w = Warehouse(_inv(n=2, stock=2.0, rop=2.0, batch=5.0, lt=2.0))
    env = SimEnvironment(dt=1.0, end=5.0)
    env.register(w)
    env.run()
    # After 5 ticks both SKUs should have received their batch.
    assert np.all(w.inv.stock_level >= 2.0 + 5.0 - 1e-9)


def test_has_work_reflects_outstanding_orders():
    w = Warehouse(_inv(n=1, stock=10.0, rop=2.0))
    env = SimEnvironment(dt=1.0, end=5.0)
    assert not w.has_work(env)
    w.inv.stock_level[0] = 1.0
    w.process_orders(elapsed=1.0)
    assert w.has_work(env)
