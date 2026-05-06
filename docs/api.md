# API reference

Full API documentation lives in each module guide page, generated from
the live source tree by [mkdocstrings](https://mkdocstrings.github.io/).
Use the table below to jump directly to the reference section for the
subpackage you need.

If you are new to the library, start with the
[module guide](modules/index.md) — each page opens with a narrative
introduction before the class/function listings.

---

| Subpackage | API reference | Key classes / functions |
|---|---|---|
| `simweave.core` | [Core](modules/core.md) | `SimEnvironment`, `Entity`, `Clock`, `SimTimeAxis` |
| `simweave.continuous` | [Continuous](modules/continuous.md) | `simulate`, `MassSpringDamper`, `QuarterCarModel`, `FullCarModel` |
| `simweave.discrete` | [Discrete](modules/discrete.md) | `Queue`, `Service`, `ArrivalGenerator`, `ResourcePool` |
| `simweave.spatial` + `.agents` | [Agents and spatial](modules/agents.md) | `Agent`, `a_star`, `dijkstra`, `grid_graph` |
| `simweave.supplychain` | [Supply chain](modules/supplychain.md) | `Warehouse`, `InventoryItems` |
| `simweave.reliability` | [Reliability](modules/reliability.md) | `ReliableEntity`, `Fleet`, `RepairCentre`, `sensitivity_sweep` |
| `simweave.roads` | [Roads](modules/roads.md) | `Road`, `Intersection`, `Roundabout`, `TrafficSignal`, `RoadNetwork` |
| `simweave.mc` | [Monte Carlo](modules/mc.md) | `run_monte_carlo`, `run_batched_mc`, `MCResult` |
| `simweave.currency` | [Currency](modules/currency.md) | `Money`, `FXConverter`, `format_money` |
| `simweave.units` | [Units](modules/units.md) | `SIUnit`, `Distance`, `Velocity`, `Mass` |
| `simweave.viz` | [Visualisation](modules/viz.md) | `plot_*` helpers, recorders, themes |
