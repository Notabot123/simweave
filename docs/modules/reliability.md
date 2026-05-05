# Reliability

`simweave.reliability` provides an availability and maintainability simulation
framework built on top of the existing discrete-event and supply-chain
primitives.  It is designed for scenarios where the *operational availability*
of a fleet of assets -- taxis, military vehicles, aircraft, industrial machines
-- is a key performance indicator driven by:

* failure rates of individual subsystems,
* spare parts holdings in one or more warehouses,
* repair-bay capacity and technician resources, and
* the financial cost of new buys vs. repairs.

---

## Concepts

### Subsystem

A **subsystem** is any replaceable or repairable component fitted to an asset
(engine, gearbox, tyre set, avionics module, etc.).  Each subsystem is
described by a [`SubsystemSpec`][simweave.reliability.SubsystemSpec] and can
be either:

| Type | On failure |
|---|---|
| **Consumable** | Failed unit discarded; new unit drawn from warehouse stock |
| **Repairable** | Failed unit sent to a [`RepairCentre`][simweave.reliability.RepairCentre] for repair |
| **Beyond economic repair (BER)** | A fraction of repairable failures that are uneconomical to fix → treated as new buy |

Failure events follow an **exponential (memoryless) distribution**, the
standard model for electronic and mechanical components in the absence of
wear-out.  Both *time-based* (`failure_rate` in failures/day) and
*cycle-based* (`failure_rate_per_cycle` in failures/km, sortie, etc.) failure
rates can be active simultaneously.

### ReliableEntity

A [`ReliableEntity`][simweave.reliability.ReliableEntity] inherits from
[`Entity`][simweave.core.Entity] and owns a list of
[`SubsystemSpec`][simweave.reliability.SubsystemSpec] objects.  On every
simulation tick it:

1. Checks each **UP** subsystem for a random failure event.
2. For newly failed subsystems, draws a spare part from the linked
   [`Warehouse`][simweave.supplychain.Warehouse].
3. Retries **AWAITING_PART** subsystems every tick until stock is replenished.
4. Submits a [`RepairJob`][simweave.reliability.RepairJob] to the
   [`RepairCentre`][simweave.reliability.RepairCentre] once parts are in hand.
5. Tracks cumulative operational time, downtime, and costs.

An entity is **operational** only when *all* of its subsystems are UP.

### RepairCentre

[`RepairCentre`][simweave.reliability.RepairCentre] is a subclass of
[`Service`][simweave.discrete.Service].  It inherits all of `Service`'s
queuing and multi-channel machinery.  Model a repair team under the
operator's employment by passing a
[`ResourcePool`][simweave.discrete.ResourcePool] of technicians.  For a
third-party maintenance contract simply tune `capacity` and `buffer_size` to
reflect the contracted throughput.

On each job completion the `RepairCentre`:

- Returns repaired units to warehouse stock (repairable, non-BER cases).
- Records cost and counters (`total_newbuys`, `total_repairs`, `total_cost`).
- Calls back into the owning `ReliableEntity` to restore the subsystem to UP.

### Fleet and FleetAvailabilityRecorder

[`Fleet`][simweave.reliability.Fleet] is a thin wrapper around a list of
`ReliableEntity` instances with aggregate properties:

| Property | Description |
|---|---|
| `operational_count` | Entities fully operational right now |
| `operational_availability` | Fraction of fleet operational right now |
| `mean_availability` | Mean of each entity's time-based empirical Ao |
| `total_cost` | Sum of new-buy + repair costs across the fleet |

[`FleetAvailabilityRecorder`][simweave.reliability.FleetAvailabilityRecorder]
is registered with the environment and snapshots fleet state each tick.
Its `times`, `operational`, `in_repair`, and `awaiting_part` lists are
fed directly to
[`plot_fleet_availability`][simweave.viz.plot_fleet_availability].

---

## Sensitivity Analysis

[`sensitivity_sweep`][simweave.reliability.sensitivity_sweep] varies one or
two scalar parameters of a *scenario builder* function across a grid and
collects a scalar metric (e.g. Ao) from each cell.  Monte Carlo averaging is
supported via the `n_runs` argument.

```python
from simweave.reliability import sensitivity_sweep

def build(n_bays, stock_mult, seed):
    # ... build and run scenario ...
    return operational_availability   # scalar

result = sensitivity_sweep(
    build,
    param1_name="repair_bays",
    param1_values=[1, 2, 3, 4],
    param2_name="stock_multiplier",
    param2_values=[0.5, 1.0, 1.5, 2.0],
    metric_name="Ao",
    n_runs=30,
)
```

The [`SweepResult`][simweave.reliability.SweepResult] can be passed to
[`plot_sensitivity_surface`][simweave.viz.plot_sensitivity_surface] for a 3-D
surface, heatmap, or grouped bar chart.

---

## Quick start

```python
import numpy as np
import simweave as sw

# 1. Describe subsystems
specs = [
    sw.SubsystemSpec(
        name="engine",
        failure_rate=1/120,      # MTBF = 120 days
        sku_index=0,
        consumable=False,
        beyond_economic_repair_prc=0.10,
        repair_time=5.0,
        unit_cost=8_000.0,
        repair_cost=2_500.0,
    ),
    sw.SubsystemSpec(
        name="tyres",
        failure_rate=1/45,
        sku_index=1,
        consumable=True,
        repair_time=0.5,
        unit_cost=400.0,
    ),
]

# 2. Build warehouse
inv = sw.InventoryItems(
    part_names=["engine", "tyres"],
    unit_cost=[8_000.0, 400.0],
    stock_level=[3.0, 10.0],
    batchsize=[2.0, 4.0],
    reorder_points=[1.0, 2.0],
    repairable_prc=[0.90, 0.0],
    repair_times=[5.0, 0.0],
    newbuy_leadtimes=[14.0, 3.0],
)
warehouse = sw.Warehouse(inventory=inv, name="depot")

# 3. Build repair centre (2 bays, 3 technicians)
technicians = sw.ResourcePool(maxlen=3, name="technicians")
for i in range(3):
    technicians.deposit(sw.Resource(name=f"tech_{i}"))
repair_centre = sw.RepairCentre(capacity=2, resources=technicians)

# 4. Build fleet
rng = np.random.default_rng(42)
vehicles = [
    sw.ReliableEntity(
        subsystems=specs,
        warehouse=warehouse,
        repair_centre=repair_centre,
        name=f"taxi_{i:02d}",
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    for i in range(10)
]
fleet = sw.Fleet(vehicles, name="taxi_fleet")
recorder = sw.FleetAvailabilityRecorder(fleet)

# 5. Run
env = sw.SimEnvironment(dt=1.0, end=365.0)
env.register(warehouse)
env.register(repair_centre)
for v in vehicles:
    env.register(v)
env.register(recorder)
env.run(until=365.0)

# 6. Summarise
print(f"Operational availability: {recorder.mean_operational_availability:.3f}")
print(f"Total fleet cost: £{fleet.total_cost:,.0f}")

# 7. Plot
fig = sw.plot_fleet_availability(recorder, title="Taxi Fleet Availability")
fig.show()
```

---

## API reference

::: simweave.reliability.SubsystemSpec

::: simweave.reliability.SubsystemState

::: simweave.reliability.SubsystemStatus

::: simweave.reliability.ReliableEntity

::: simweave.reliability.RepairJob

::: simweave.reliability.RepairCentre

::: simweave.reliability.Fleet

::: simweave.reliability.FleetAvailabilityRecorder

::: simweave.reliability.SweepResult

::: simweave.reliability.sensitivity_sweep
