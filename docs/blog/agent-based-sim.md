# SimWeave vs Mesa: From Boltzmann Wealth to Full‑Stack Logistics Simulation

*A practical comparison using the classic Boltzmann Wealth Model, with various extensions and warehouse inventory optimisation.*

Mesa’s Boltzmann Wealth Model is a classic demonstration of emergent behaviour in agent‑based modelling. It shows how simple random exchanges between agents can produce an exponential wealth distribution.

## Table of Contents
1. [Introduction](#introduction)
2. [Baseline: Mesa‑Style Wealth Exchange](#baseline-mesa-style-wealth-exchange)
3. [Adding Movement and Routing](#adding-movement-and-routing)
4. [Adding Warehouses and Inventory](#adding-warehouses-and-inventory)
5. [Adding a Central Depot with Queueing](#adding-a-central-depot-with-queueing)
6. [Event‑Driven Stochastic Processes](#event-driven-stochastic-processes)
7. [Optimisation: Cost vs Availability](#optimisation-cost-vs-availability)
8. [Visualisation](#visualisation)
9. [Conclusion](#conclusion)
10. [Try it out yourself](#try-it-out-yourself)

---

## Introduction

SimWeave can reproduce this model, but it can also extend it far beyond what Mesa was designed for. In this article, we begin with the familiar Mesa‑style wealth model and gradually evolve it into a full logistics simulation involving continuous time, routing, warehouses, queueing, and optimisation.

## Baseline: Mesa‑Style Wealth Exchange

We begin with a simple model:

- Each trader has a scalar wealth value.
- At random times, two traders exchange a fraction of wealth.
- Over time, the distribution becomes exponential‑like.

SimWeave implements this using:

- a continuous‑time simulation environment,
- a tick‑driven entity that generates random meetings,
- exponential inter‑arrival times.

This reproduces the behaviour of the Mesa model while operating in continuous time.

---

## Adding Movement and Routing

Mesa’s agents typically move on a discrete grid with synchronous steps.  
SimWeave’s `Agent` class provides:

- continuous movement based on speed and time step,
- A* routing on arbitrary graphs,
- arrival callbacks,
- shared graphs between agents.

We place traders on an 8×8 grid and give them random waypoints.  
Wealth exchanges now occur only when agents meet at the same node.  
This introduces spatial structure and routing dynamics that Mesa does not provide out of the box.

---

## Adding Warehouses and Inventory

Each trader is now given a `Warehouse` with:

- stock levels,
- batch sizes,
- reorder points,
- lead times,
- automatic reorder logic.

Trades consume stock.  
Warehouses reorder when needed.  
Lead times and replenishment are handled automatically in `Warehouse.tick()`.

This transforms the model from a pure ABM into a supply chain simulation.

---

## Adding a Central Depot with Queueing

We introduce a central depot located at the centre of the grid.  
It is modelled as a `Service` with:

- a pre‑service queue,
- a single processing channel,
- configurable service times.

Traders travel to the depot when their stock falls below a threshold.  
They queue, wait for service, receive replenishment, and return to their route.

This demonstrates SimWeave’s queueing and service abstractions, which allow realistic modelling of congestion, waiting times, and resource contention.

---

## Event‑Driven Stochastic Processes

SimWeave supports both:

- tick‑driven entities (agents, warehouses, services), and
- event‑driven callbacks (random meetings, scheduled events).

This hybrid model allows:

- continuous movement,
- sparse stochastic events,
- fast‑forwarding through idle periods.

This is a significant modelling advantage over frameworks that rely solely on discrete ticks or coroutine‑based processes.

---

## Optimisation: Cost vs Availability

After running the simulation, we estimate demand using:

```
warehouse.estimate_demand_rate(total_time)
```

We then run a Pareto sweep to explore the trade‑off between cost and availability.  
SimWeave provides:

- a Poisson heuristic,
- a differential evolution optimiser,
- a Plotly‑based visualisation of the Pareto frontier.

This connects simulation outputs directly to optimisation, enabling data‑driven decision making.

---

## Visualisation

SimWeave integrates with Plotly for:

- agent trajectories,
- wealth distributions,
- queue lengths,
- depot utilisation,
- cost vs availability frontiers.

Animations of agent movement can be generated using a helper function that exports frames and assembles them into a GIF.

---

## Conclusion

Mesa is an excellent framework for discrete, grid‑based agent models.  
SimWeave extends this paradigm into:

- continuous time,
- routing and spatial networks,
- inventory and supply chains,
- queueing and service systems,
- optimisation.

This example demonstrates how a simple wealth exchange model can evolve into a full logistics simulation within a single coherent framework.

## Try it out yourself

Full runnable code is in the [companion notebook](https://github.com/Notabot123/simweave-notebooks).
