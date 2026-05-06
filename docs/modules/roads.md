# Road networks

`simweave.roads` provides discrete-event road-network primitives that sit
naturally alongside queuing systems and supply chains inside a shared
`SimEnvironment`.  Build anything from a single signalised crossroads to a
multi-junction urban network.

The two charts below come from demos 24 and 25.  The left chart shows
approach-road occupancy at a 4-way signalised junction; the right shows a
UK roundabout's circulating and arm-queue vehicle counts over a one-hour
simulation.

<iframe src="../../embeds/intersection_occupancy.html"
        width="100%" height="420" frameborder="0"
        loading="lazy"
        title="Signalised crossroads approach occupancy"></iframe>

<iframe src="../../embeds/roundabout_occupancy.html"
        width="100%" height="420" frameborder="0"
        loading="lazy"
        title="Roundabout circulating and queued vehicles"></iframe>

---

## Concepts

### Road

A [`Road`][simweave.roads.Road] is a **free-flow conveyor belt**.  When a
vehicle calls `road.enter(vehicle, env)` a delivery event is scheduled
`length / effective_speed` simulation-time units in the future.  Multiple
vehicles travel simultaneously — there is no blocking or overtaking in this
model.

Travel time is governed by the road's `speed_limit`.  Vehicles may carry an
own-speed override that is capped at the limit (a slow lorry is slower; a
fast sports car is no quicker than the limit).

[`DualCarriageway`][simweave.roads.DualCarriageway] is a convenience wrapper
around two opposing `Road` instances sharing the same corridor.

### Vehicle

[`Vehicle`][simweave.roads.Vehicle] is a thin
[`Entity`][simweave.core.Entity] that carries an optional speed override and
accumulates travel statistics (`roads_traversed`, `total_travel_time`).

[`VehicleArrivalProcess`][simweave.roads.VehicleArrivalProcess] generates
`Vehicle` instances at a given inter-arrival distribution and enters them
onto a target road — analogous to `ArrivalGenerator` in the discrete
queueing module.

### TrafficSignal and SignalPhase

[`TrafficSignal`][simweave.roads.TrafficSignal] cycles through a list of
[`SignalPhase`][simweave.roads.SignalPhase] objects.  Each phase specifies
which approach roads are green and how long the phase lasts.  The signal
is a registered process: it ticks and advances its phase timer each
simulation step.

**Important registration order**: always register a `TrafficSignal`
*before* any `Intersection` it controls.  The signal must update its phase
state before the intersection dispatches queued vehicles.

```python
env.register(signal)       # 1 — signal ticks first
env.register(intersection) # 2 — then intersection releases queued vehicles
```

`RoadNetwork.register_all()` handles this automatically.

### Intersection

[`Intersection`][simweave.roads.Intersection] holds a FIFO approach queue
per incoming road and a weighted list of exit roads.  On each tick it
processes one vehicle per green approach, routing it to an exit road by
weighted random selection.

Without a `TrafficSignal` the intersection acts as a **give-way / all-way
stop**: all approaches are permanently green and vehicles pass through as
fast as they arrive.

```python
junction = sw.Intersection(signal=signal, name="high_st_junction")
junction.add_approach(road_north)
junction.add_approach(road_south)
junction.add_exit(exit_east, weight=2.0)   # twice as likely as west
junction.add_exit(exit_west, weight=1.0)
road_north.outlet = junction
road_south.outlet = junction
```

### Roundabout

[`Roundabout`][simweave.roads.Roundabout] implements a
**priority-to-circulating-traffic** model.  Vehicles arriving at an arm
entry wait if the roundabout is at capacity (`max_circulating`).  Once
admitted, they spend `transit_time` simulation-time units circulating and
are then routed to an exit road.

[`Handedness`][simweave.roads.Handedness] records the driving-side
convention:

| Value | Circulation | Countries |
|---|---|---|
| `Handedness.LEFT` | Clockwise (from above) | UK, Australia, Japan … |
| `Handedness.RIGHT` | Anti-clockwise | EU, US, most of the world |

```python
rb = sw.Roundabout(
    max_circulating=6,
    transit_time=6.0,          # seconds inside the roundabout
    handedness=sw.Handedness.LEFT,
    name="town_centre_roundabout",
)
rb.add_entry(road_north)
rb.add_exit(exit_east,  weight=1.0)
rb.add_exit(exit_south, weight=1.0)
rb.add_exit(exit_west,  weight=1.0)
road_north.outlet = rb
```

### RoadNetwork

[`RoadNetwork`][simweave.roads.RoadNetwork] is a container that registers
all components with a `SimEnvironment` in the correct tick order via a
single `register_all(env)` call.

---

## Quick start

### Signalised junction

```python
import numpy as np
import simweave as sw

# ── Road geometry ──────────────────────────────────────────────────────────
road_ns = sw.Road(200.0, 13.9, lanes=2, name="NS_approach")
road_ew = sw.Road(200.0, 13.9, lanes=2, name="EW_approach")
exit_ns = sw.Road(200.0, 13.9, name="NS_exit")
exit_ew = sw.Road(200.0, 13.9, name="EW_exit")

# ── Traffic signal (45 s NS / 30 s EW) ────────────────────────────────────
signal = sw.TrafficSignal([
    sw.SignalPhase(green_roads=[road_ns], duration=45.0, name="NS_green"),
    sw.SignalPhase(green_roads=[road_ew], duration=30.0, name="EW_green"),
])

# ── Intersection ───────────────────────────────────────────────────────────
junction = sw.Intersection(signal=signal, name="junction")
for road in (road_ns, road_ew):
    junction.add_approach(road)
    road.outlet = junction
junction.add_exit(exit_ns, weight=1.0)
junction.add_exit(exit_ew, weight=1.0)

# ── Arrival processes ──────────────────────────────────────────────────────
rng = np.random.default_rng(0)
arr_ns = sw.VehicleArrivalProcess(
    interarrival=lambda r: r.exponential(5.0),   # 1 vehicle / 5 s
    road=road_ns, rng=rng,
)
arr_ew = sw.VehicleArrivalProcess(
    interarrival=lambda r: r.exponential(4.0),
    road=road_ew, rng=rng,
)

# ── Recorders ──────────────────────────────────────────────────────────────
occ_rec = sw.RoadOccupancyRecorder([road_ns, road_ew])
q_rec   = sw.IntersectionQueueRecorder(junction)

# ── Assemble and run ───────────────────────────────────────────────────────
net = sw.RoadNetwork()
net.add_signal(signal)
net.add_intersection(junction)
for r in (road_ns, road_ew, exit_ns, exit_ew):
    net.add_road(r)
net.add_arrival_process(arr_ns)
net.add_arrival_process(arr_ew)
net.add_recorder(occ_rec)
net.add_recorder(q_rec)

env = sw.SimEnvironment(dt=1.0, end=3600.0)
net.register_all(env)
env.run()

print(f"Cleared  : {junction.total_vehicles:,}")
print(f"Delayed  : {junction.total_delayed:,}")

fig = sw.plot_intersection_queues(q_rec, title="Junction queue lengths")
fig.show()
```

### Roundabout

```python
rb = sw.Roundabout(
    max_circulating=6,
    transit_time=6.0,
    handedness=sw.Handedness.LEFT,
    name="roundabout",
)
rb.add_entry(road_north)
rb.add_entry(road_south)
rb.add_entry(road_east)
rb.add_entry(road_west)
rb.add_exit(exit_north, weight=1.0)
rb.add_exit(exit_south, weight=1.0)
rb.add_exit(exit_east,  weight=1.0)
rb.add_exit(exit_west,  weight=1.0)
for road in (road_north, road_south, road_east, road_west):
    road.outlet = rb
```

---

## API reference

::: simweave.roads.Vehicle

::: simweave.roads.VehicleArrivalProcess

::: simweave.roads.Road

::: simweave.roads.DualCarriageway

::: simweave.roads.SignalPhase

::: simweave.roads.TrafficSignal

::: simweave.roads.Intersection

::: simweave.roads.Handedness

::: simweave.roads.Roundabout

::: simweave.roads.RoadNetwork

::: simweave.roads.RoadOccupancyRecorder

::: simweave.roads.IntersectionQueueRecorder
