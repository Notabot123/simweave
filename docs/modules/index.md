# Modules

SimWeave is organised into focused subpackages. Most users will only
ever import from the top-level `simweave` namespace, which re-exports
the primitives below.

| Subpackage | Page | Purpose |
| --- | --- | --- |
| `simweave.core` | [Core](core.md) | `SimEnvironment`, `Entity`, clock, event queue, `SimTimeAxis` |
| `simweave.continuous` | [Continuous](continuous.md) | ODE integration, canonical example systems |
| `simweave.discrete` | [Discrete](discrete.md) | Queues, services, generators, RNG distributions |
| `simweave.spatial` + `.agents` | [Agents and spatial](agents.md) | Graphs, A\* / Dijkstra, agent traversal |
| `simweave.supplychain` | [Supply chain](supplychain.md) | Warehouses, multi-SKU inventory |
| `simweave.reliability` | [Reliability](reliability.md) | Fleet availability, failure rates, repair centres, sensitivity sweeps |
| `simweave.roads` | [Roads](roads.md) | Roads, intersections, roundabouts, traffic signals |
| `simweave.mc` | [Monte Carlo](mc.md) | Replication, batching, `MCResult` |
| `simweave.currency` | [Currency](currency.md) | `Money`, FX conversion, locale formatting |
| `simweave.units` | [Units](units.md) | SI quantity helpers |
| `simweave.viz` | [Visualisation](viz.md) | Plotly plot helpers, themes, recorders |

For a quick lookup by class or function name, see the
[API reference](../api.md).
