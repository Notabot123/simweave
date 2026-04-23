# Modules

SimWeave is organised into focused subpackages. Most users will only
ever import from the top-level `simweave` namespace, which re-exports
the primitives below.

| Subpackage              | Pages                                           | Purpose                                              |
| ----------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| `simweave.core`         | [Core](core.md)                                 | `SimEnvironment`, `Entity`, clock, event queue       |
| `simweave.continuous`   | [Continuous](continuous.md)                     | ODE integration, canonical example systems           |
| `simweave.discrete`     | [Discrete](discrete.md)                         | Queues, services, generators, RNG distributions      |
| `simweave.spatial` + `.agents` | [Agents and spatial](agents.md)          | Graphs, A\* / Dijkstra, agent traversal              |
| `simweave.supplychain`  | [Supply chain](supplychain.md)                  | Warehouses, multi-SKU inventory                      |
| `simweave.mc`           | [Monte Carlo](mc.md)                            | Replication, batching, `MCResult`                    |
| `simweave.currency`     | [Currency](currency.md)                         | `Money`, FX conversion, locale formatting            |
| `simweave.units`        | [Units](units.md)                               | SI quantity helpers                                  |
| `simweave.viz`          | [Visualisation](viz.md)                         | Plotly plot helpers, themes, recorders               |

For the fully-rendered import graph, see the [API reference](../api.md).
