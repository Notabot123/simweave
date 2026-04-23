# Supply chain

Multi-SKU warehouses and the inventory record they hold.

## `InventoryItems`

A dataclass capturing the per-SKU attributes a warehouse needs:

```python
inv = sw.InventoryItems(
    part_names=["widget", "gizmo", "sprocket"],
    unit_cost=[1.0, 2.5, 0.7],
    stock_level=[20.0, 10.0, 50.0],
    batchsize=[20.0, 10.0, 50.0],
    reorder_points=[5.0, 3.0, 15.0],
    repairable_prc=[0.0, 0.0, 0.0],
    repair_times=[1.0, 1.0, 1.0],
    newbuy_leadtimes=[3.0, 5.0, 2.0],
)
```

All list-shaped fields must be the same length. Numeric fields are
coerced to `np.ndarray` after construction so warehouse operations stay
vectorised.

## `Warehouse`

```python
wh = sw.Warehouse(inventory=inv, name="store")

env = sw.SimEnvironment(dt=1.0, end=80.0)
env.register(wh)
env.run()
```

Decrement stock during a tick using `wh.decrement_vector(per_sku_amounts)`.
The warehouse handles the reorder logic against `inventory.reorder_points`.

## Recording stock for plots

```python
rec = sw.WarehouseStockRecorder(wh, name="store_stock")
env.register(rec)        # AFTER the warehouse so the recorder snapshots
                         # post-tick state
...
sw.plot_warehouse_stock(rec).write_html("stock.html",
                                        include_plotlyjs="cdn")
```

<iframe src="../../embeds/warehouse_stock.html"
        width="100%" height="500" frameborder="0"
        loading="lazy"
        title="Warehouse stock vs reorder points"></iframe>

## API

::: simweave.supplychain
    options:
      show_root_heading: false
      show_source: true
