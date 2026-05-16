import simweave as sw
import numpy as np

def warehouse_with_demand(n=3, rate=1.0):
    inv = sw.InventoryItems(
        part_names=[f"sku_{i}" for i in range(n)],
        unit_cost=[1.0 + i for i in range(n)],
        stock_level=[10.0] * n,
        batchsize=[5.0] * n,
        reorder_points=[2.0] * n,
        repairable_prc=[0.0] * n,
        repair_times=[0.0] * n,
        newbuy_leadtimes=[3.0] * n,
    )
    w = sw.Warehouse(inv, name="w")
    w._demand_rate = np.full(n, rate)
    return w