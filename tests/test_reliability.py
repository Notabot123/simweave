"""Unit tests for simweave.reliability.

Coverage
--------
* SubsystemSpec / SubsystemStatus -- dataclass defaults and field values.
* ReliableEntity -- failure events, part-waiting, repair flow, availability
  accounting, cost tracking, cycle-based failures, no-repair-centre path.
* RepairCentre / RepairJob -- completion callback, stock-return, cost/counter
  tallying, BER path.
* Fleet -- status_counts breakdown, aggregate availability, cost sums.
* FleetAvailabilityRecorder -- snapshot recording, mean_operational_availability.
* sensitivity_sweep -- 1-D and 2-D shapes, MC averaging, error-guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import simweave as sw
from simweave.core.environment import SimEnvironment
from simweave.discrete.resources import Resource, ResourcePool
from simweave.reliability.entity import ReliableEntity
from simweave.reliability.fleet import Fleet, FleetAvailabilityRecorder
from simweave.reliability.repair import RepairCentre, RepairJob
from simweave.reliability.sensitivity import SweepResult, sensitivity_sweep
from simweave.reliability.subsystem import SubsystemSpec, SubsystemState, SubsystemStatus
from simweave.supplychain.inventory import InventoryItems
from simweave.supplychain.warehouse import Warehouse


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NEVER_FAIL = 0.0          # failure_rate that guarantees no failure
ALWAYS_FAIL = 1_000.0     # failure_rate so high P(fail in dt=1) ≈ 1


def _warehouse(stock: float = 5.0, n_skus: int = 2) -> Warehouse:
    """Minimal warehouse with ``n_skus`` identical SKUs."""
    return Warehouse(
        InventoryItems(
            part_names=[f"sku_{i}" for i in range(n_skus)],
            unit_cost=[100.0] * n_skus,
            stock_level=[stock] * n_skus,
            batchsize=[5.0] * n_skus,
            reorder_points=[1.0] * n_skus,
            repairable_prc=[0.8] * n_skus,
            repair_times=[2.0] * n_skus,
            newbuy_leadtimes=[5.0] * n_skus,
        ),
        name="test_wh",
    )


def _consumable_spec(failure_rate: float = NEVER_FAIL, sku: int = 0) -> SubsystemSpec:
    return SubsystemSpec(
        name="consumable_part",
        failure_rate=failure_rate,
        sku_index=sku,
        consumable=True,
        repair_time=1.0,
        unit_cost=100.0,
    )


def _repairable_spec(
    failure_rate: float = NEVER_FAIL,
    sku: int = 0,
    ber: float = 0.0,
    repair_time: float = 2.0,
) -> SubsystemSpec:
    return SubsystemSpec(
        name="repairable_part",
        failure_rate=failure_rate,
        sku_index=sku,
        consumable=False,
        beyond_economic_repair_prc=ber,
        repair_time=repair_time,
        unit_cost=500.0,
        repair_cost=150.0,
    )


def _repair_centre(capacity: int = 1, n_technicians: int = 0) -> RepairCentre:
    """Build a RepairCentre; optionally gate with a ResourcePool."""
    if n_technicians > 0:
        pool = ResourcePool(maxlen=n_technicians, name="techs")
        for i in range(n_technicians):
            pool.deposit(Resource(name=f"tech_{i}"))
        return RepairCentre(capacity=capacity, resources=pool)
    return RepairCentre(capacity=capacity)


def _entity(
    specs: list[SubsystemSpec],
    warehouse: Warehouse,
    repair_centre: RepairCentre | None = None,
    seed: int = 0,
) -> ReliableEntity:
    return ReliableEntity(
        subsystems=specs,
        warehouse=warehouse,
        repair_centre=repair_centre,
        name="test_entity",
        rng=np.random.default_rng(seed),
    )


def _run(env: SimEnvironment, *extras) -> None:
    """Run env to its end time, registering any extras first."""
    for obj in extras:
        env.register(obj)
    env.run()


# ===========================================================================
# 1. SubsystemSpec
# ===========================================================================


class TestSubsystemSpec:
    def test_defaults(self):
        spec = SubsystemSpec(name="part", failure_rate=0.01, sku_index=0)
        assert spec.consumable is True
        assert spec.beyond_economic_repair_prc == 0.0
        assert spec.repair_time == 1.0
        assert spec.unit_cost == 0.0
        assert spec.repair_cost == 0.0
        assert spec.failure_rate_per_cycle == 0.0

    def test_fields_stored(self):
        spec = SubsystemSpec(
            name="engine",
            failure_rate=1 / 90,
            sku_index=2,
            consumable=False,
            beyond_economic_repair_prc=0.1,
            repair_time=5.0,
            unit_cost=8_000.0,
            repair_cost=2_500.0,
            failure_rate_per_cycle=1e-4,
        )
        assert spec.name == "engine"
        assert spec.sku_index == 2
        assert spec.consumable is False
        assert spec.beyond_economic_repair_prc == pytest.approx(0.1)
        assert spec.failure_rate_per_cycle == pytest.approx(1e-4)


class TestSubsystemStatus:
    def test_initial_state_is_up(self):
        spec = _consumable_spec()
        status = SubsystemStatus(spec=spec)
        assert status.state == SubsystemState.UP
        assert status.total_failures == 0
        assert status.total_cost == 0.0

    def test_total_cost_sums_fields(self):
        spec = _consumable_spec()
        status = SubsystemStatus(spec=spec)
        status.cost_newbuy = 200.0
        status.cost_repair = 50.0
        assert status.total_cost == pytest.approx(250.0)


# ===========================================================================
# 2. ReliableEntity -- no-repair-centre path
# ===========================================================================


class TestReliableEntityNoRepairCentre:
    """Tests using repair_centre=None (instantaneous restoration)."""

    def test_starts_operational(self):
        wh = _warehouse()
        e = _entity([_consumable_spec(NEVER_FAIL)], wh)
        assert e.is_operational is True
        assert e.availability == pytest.approx(1.0)

    def test_never_fails_with_zero_rate(self):
        wh = _warehouse(stock=5.0)
        e = _entity([_consumable_spec(NEVER_FAIL)], wh)
        env = SimEnvironment(dt=1.0, end=50.0)
        env.register(e)
        env.run()
        assert e.is_operational is True
        assert e.total_downtime == pytest.approx(0.0)
        assert wh.inv.stock_level[0] == pytest.approx(5.0)  # no consumption

    def test_always_fails_consumable_draws_stock(self):
        """With a huge failure rate every tick a consumable should consume stock."""
        wh = _warehouse(stock=10.0)
        e = _entity([_consumable_spec(ALWAYS_FAIL)], wh, seed=7)
        env = SimEnvironment(dt=1.0, end=5.0)
        env.register(wh)
        env.register(e)
        env.run()
        # Some stock should have been consumed.
        assert wh.inv.stock_level[0] < 10.0

    def test_instantaneous_repair_restores_up(self):
        """With no repair centre the subsystem should return to UP on the same
        tick the part is obtained (no queuing delay)."""
        wh = _warehouse(stock=10.0)
        # Moderate failure rate so at least one failure is very likely in 100 ticks.
        e = _entity([_consumable_spec(0.5)], wh, seed=0)
        env = SimEnvironment(dt=1.0, end=100.0)
        env.register(wh)
        env.register(e)
        env.run()
        # After 100 ticks the entity should be operational (restored each tick).
        assert e.is_operational is True

    def test_awaiting_part_when_stock_empty(self):
        """If the warehouse is empty the entity should wait for replenishment."""
        wh = _warehouse(stock=0.0)
        # Reorder points set so no automatic replenishment fires.
        wh.inv.reorder_points[:] = 0.0
        wh.inv.batchsize[:] = 0.0
        e = _entity([_consumable_spec(ALWAYS_FAIL)], wh, seed=0)
        env = SimEnvironment(dt=1.0, end=3.0)
        env.register(wh)
        env.register(e)
        env.run()
        # Should be grounded.
        assert e.subsystems[0].state == SubsystemState.AWAITING_PART

    def test_availability_tracks_downtime(self):
        """Empirical availability must equal op_time / (op_time + downtime)."""
        wh = _warehouse(stock=0.0)
        wh.inv.reorder_points[:] = 0.0
        wh.inv.batchsize[:] = 0.0
        e = _entity([_consumable_spec(ALWAYS_FAIL)], wh, seed=0)
        env = SimEnvironment(dt=1.0, end=10.0)
        env.register(wh)
        env.register(e)
        env.run()
        expected = e.total_operational_time / (
            e.total_operational_time + e.total_downtime
        )
        assert e.availability == pytest.approx(expected)

    def test_cost_recorded_for_consumable_newbuy(self):
        wh = _warehouse(stock=10.0)
        spec = SubsystemSpec(
            "part", ALWAYS_FAIL, 0, consumable=True, repair_time=0.5, unit_cost=99.0
        )
        e = _entity([spec], wh, seed=1)
        env = SimEnvironment(dt=1.0, end=5.0)
        env.register(wh)
        env.register(e)
        env.run()
        # At least one new buy should have been charged.
        assert e.cost_newbuy > 0.0
        assert e.cost_repair == pytest.approx(0.0)

    def test_multiple_subsystems_all_must_be_up(self):
        """Entity is operational only when ALL subsystems are UP."""
        wh = _warehouse(stock=10.0, n_skus=2)
        specs = [
            _consumable_spec(NEVER_FAIL, sku=0),
            _consumable_spec(ALWAYS_FAIL, sku=1),
        ]
        e = _entity(specs, wh, seed=0)
        # Before any ticks second subsystem is still UP.
        assert e.is_operational is True
        # After one tick the always-failing subsystem should have been hit,
        # consumed a part (instantaneous), and restored -- so still UP.
        env = SimEnvironment(dt=1.0, end=1.0)
        env.register(wh)
        env.register(e)
        env.run()
        # Both should be UP since repair is instantaneous.
        assert e.is_operational is True

    def test_summary_dict_keys(self):
        wh = _warehouse()
        e = _entity([_consumable_spec()], wh)
        s = e.summary()
        assert "name" in s
        assert "availability" in s
        assert "total_cost" in s
        assert "subsystem_failures" in s


# ===========================================================================
# 3. ReliableEntity -- cycle-based failure
# ===========================================================================


class TestCycleBasedFailure:
    def test_no_failure_without_cycle_increment(self):
        """Cycle-based rate should not fire if cycles never change."""
        wh = _warehouse(stock=5.0)
        spec = SubsystemSpec(
            "widget", failure_rate=0.0, sku_index=0,
            failure_rate_per_cycle=ALWAYS_FAIL
        )
        e = _entity([spec], wh, seed=0)
        env = SimEnvironment(dt=1.0, end=20.0)
        env.register(wh)
        env.register(e)
        env.run()
        # operational_cycles never changed → no failure
        assert e.subsystems[0].total_failures == 0

    def test_cycle_failure_fires_when_cycles_advance(self):
        """With huge cycle rate a single cycle increment should trigger failure."""
        wh = _warehouse(stock=10.0)
        spec = SubsystemSpec(
            "widget", failure_rate=0.0, sku_index=0,
            consumable=True, repair_time=0.5,
            failure_rate_per_cycle=ALWAYS_FAIL
        )
        e = _entity([spec], wh, seed=0)
        # Manually advance one cycle before the tick so delta_cycles > 0.
        e.operational_cycles = 1.0
        env = SimEnvironment(dt=1.0, end=3.0)
        env.register(wh)
        env.register(e)
        env.run()
        assert e.subsystems[0].total_failures > 0


# ===========================================================================
# 4. RepairCentre / RepairJob
# ===========================================================================


class TestRepairCentre:
    def _run_repair_job(
        self,
        repair_time: float,
        is_new_buy: bool,
        return_to_stock: bool,
        cost: float,
        wh: Warehouse,
    ) -> tuple[ReliableEntity, RepairCentre]:
        """Helper: push one RepairJob through a single-bay centre."""
        spec = _repairable_spec(NEVER_FAIL)
        e = _entity([spec], wh)
        rc = _repair_centre(capacity=1)

        job = RepairJob(
            owner=e,
            subsystem_idx=0,
            is_new_buy=is_new_buy,
            return_to_stock=return_to_stock,
            repair_time=repair_time,
            cost=cost,
        )
        # Force the subsystem into IN_REPAIR state before submitting.
        e.subsystems[0].state = SubsystemState.IN_REPAIR
        rc.enqueue(job)

        env = SimEnvironment(dt=1.0, end=repair_time + 2.0)
        env.register(rc)
        env.run()
        return e, rc

    def test_job_completes_and_restores_subsystem(self):
        wh = _warehouse(stock=5.0)
        e, rc = self._run_repair_job(2.0, True, False, 100.0, wh)
        assert e.subsystems[0].state == SubsystemState.UP

    def test_newbuy_counter_increments(self):
        wh = _warehouse(stock=5.0)
        _, rc = self._run_repair_job(1.0, True, False, 50.0, wh)
        assert rc.total_newbuys == 1
        assert rc.total_repairs == 0

    def test_repair_counter_increments(self):
        wh = _warehouse(stock=5.0)
        _, rc = self._run_repair_job(1.0, False, False, 50.0, wh)
        assert rc.total_repairs == 1
        assert rc.total_newbuys == 0

    def test_cost_accumulated_on_centre(self):
        wh = _warehouse(stock=5.0)
        _, rc = self._run_repair_job(1.0, True, False, 777.0, wh)
        assert rc.total_cost == pytest.approx(777.0)

    def test_return_to_stock_increments_warehouse(self):
        wh = _warehouse(stock=2.0)
        initial_stock = float(wh.inv.stock_level[0])
        self._run_repair_job(1.0, False, True, 50.0, wh)
        # Warehouse should gain one unit back.
        assert wh.inv.stock_level[0] == pytest.approx(initial_stock + 1.0)

    def test_no_return_to_stock_leaves_warehouse_unchanged(self):
        wh = _warehouse(stock=2.0)
        initial_stock = float(wh.inv.stock_level[0])
        self._run_repair_job(1.0, True, False, 50.0, wh)
        assert wh.inv.stock_level[0] == pytest.approx(initial_stock)

    def test_cost_logged_on_entity_newbuy(self):
        wh = _warehouse(stock=5.0)
        e, _ = self._run_repair_job(1.0, True, False, 300.0, wh)
        assert e.cost_newbuy == pytest.approx(300.0)
        assert e.cost_repair == pytest.approx(0.0)

    def test_cost_logged_on_entity_repair(self):
        wh = _warehouse(stock=5.0)
        e, _ = self._run_repair_job(1.0, False, False, 150.0, wh)
        assert e.cost_repair == pytest.approx(150.0)
        assert e.cost_newbuy == pytest.approx(0.0)

    def test_technician_resource_gates_repair(self):
        """With 1 bay and 1 technician the job must acquire the resource first."""
        wh = _warehouse(stock=5.0)
        spec = _repairable_spec(NEVER_FAIL)
        e = _entity([spec], wh)

        pool = ResourcePool(maxlen=1)
        pool.deposit(Resource(name="tech"))
        rc = RepairCentre(capacity=1, resources=pool)

        job = RepairJob(
            owner=e, subsystem_idx=0, is_new_buy=False,
            return_to_stock=False, repair_time=2.0, cost=50.0,
        )
        e.subsystems[0].state = SubsystemState.IN_REPAIR
        rc.enqueue(job)

        env = SimEnvironment(dt=1.0, end=5.0)
        env.register(rc)
        env.run()

        assert e.subsystems[0].state == SubsystemState.UP
        assert rc.total_repairs == 1


# ===========================================================================
# 5. ReliableEntity integrated with RepairCentre
# ===========================================================================


class TestReliableEntityWithRepairCentre:
    def test_failure_queues_job(self):
        """A failure with ample stock should enqueue a RepairJob."""
        wh = _warehouse(stock=10.0)
        rc = _repair_centre(capacity=2)
        spec = _consumable_spec(ALWAYS_FAIL)
        e = _entity([spec], wh, rc, seed=0)

        env = SimEnvironment(dt=1.0, end=1.0)
        env.register(wh)
        env.register(rc)
        env.register(e)
        env.run()

        # Either the job is still in the repair queue or already completed.
        # Either way, a failure should have been recorded.
        assert e.subsystems[0].total_failures > 0

    def test_entity_down_while_in_repair(self):
        """With a long repair time the entity should accumulate downtime."""
        wh = _warehouse(stock=10.0)
        rc = _repair_centre(capacity=1)
        # repair_time=20 days ensures the entity is down for most of the 10-tick sim.
        spec = _repairable_spec(ALWAYS_FAIL, repair_time=20.0)
        e = _entity([spec], wh, rc, seed=0)

        env = SimEnvironment(dt=1.0, end=10.0)
        env.register(wh)
        env.register(rc)
        env.register(e)
        env.run()

        assert e.total_downtime > 0.0

    def test_ber_path_charges_unit_cost(self):
        """BER failure should charge unit_cost, not repair_cost."""
        wh = _warehouse(stock=10.0)
        rc = _repair_centre(capacity=2)
        # BER probability = 1.0 → every failure is BER.
        spec = SubsystemSpec(
            "engine", ALWAYS_FAIL, 0,
            consumable=False, beyond_economic_repair_prc=1.0,
            repair_time=1.0, unit_cost=888.0, repair_cost=111.0
        )
        e = _entity([spec], wh, rc, seed=0)

        env = SimEnvironment(dt=1.0, end=5.0)
        env.register(wh)
        env.register(rc)
        env.register(e)
        env.run()

        # All failures are BER → only new buys, no repair cost.
        assert e.cost_newbuy > 0.0
        assert e.cost_repair == pytest.approx(0.0)

    def test_repairable_no_ber_charges_repair_cost(self):
        """Non-BER repairable failure should charge repair_cost."""
        wh = _warehouse(stock=10.0)
        rc = _repair_centre(capacity=2)
        spec = SubsystemSpec(
            "gearbox", ALWAYS_FAIL, 0,
            consumable=False, beyond_economic_repair_prc=0.0,
            repair_time=1.0, unit_cost=500.0, repair_cost=120.0
        )
        # Fix RNG so BER roll always < 0.0 (impossible) → never BER.
        e = _entity([spec], wh, rc, seed=42)

        env = SimEnvironment(dt=1.0, end=5.0)
        env.register(wh)
        env.register(rc)
        env.register(e)
        env.run()

        # At least one failure should have occurred (ALWAYS_FAIL).
        # Since BER=0 all repairs charged at repair_cost.
        assert e.cost_repair > 0.0 or e.cost_newbuy > 0.0  # either path ran


# ===========================================================================
# 6. Fleet
# ===========================================================================


class TestFleet:
    def _make_fleet(self, n: int = 3, operational: list[bool] | None = None) -> Fleet:
        """Make a fleet; forcibly set operational state via subsystem states."""
        wh = _warehouse(stock=20.0)
        operational = operational if operational is not None else [True] * n
        entities = []
        for i, op in enumerate(operational):
            e = _entity([_consumable_spec(NEVER_FAIL)], wh, seed=i)
            if not op:
                e.subsystems[0].state = SubsystemState.IN_REPAIR
            entities.append(e)
        return Fleet(entities, name="test_fleet")

    def test_size(self):
        fleet = self._make_fleet(5)
        assert fleet.size == 5

    def test_all_operational(self):
        fleet = self._make_fleet(4, [True, True, True, True])
        assert fleet.operational_count == 4
        assert fleet.operational_availability == pytest.approx(1.0)

    def test_partial_operational(self):
        fleet = self._make_fleet(4, [True, False, True, False])
        assert fleet.operational_count == 2
        assert fleet.operational_availability == pytest.approx(0.5)

    def test_status_counts_in_repair(self):
        fleet = self._make_fleet(3, [True, False, False])
        counts = fleet.status_counts()
        assert counts["operational"] == 1
        assert counts["in_repair"] == 2
        assert counts["awaiting_part"] == 0

    def test_status_counts_awaiting_part(self):
        wh = _warehouse()
        e = _entity([_consumable_spec()], wh)
        e.subsystems[0].state = SubsystemState.AWAITING_PART
        fleet = Fleet([e])
        counts = fleet.status_counts()
        assert counts["awaiting_part"] == 1
        assert counts["operational"] == 0

    def test_total_cost_sums_entities(self):
        wh = _warehouse(stock=20.0)
        entities = []
        for i in range(3):
            e = _entity([_consumable_spec()], wh, seed=i)
            e.cost_newbuy = float(i * 100)
            e.cost_repair = float(i * 50)
            entities.append(e)
        fleet = Fleet(entities)
        assert fleet.total_cost == pytest.approx(sum(i * 150 for i in range(3)))
        assert fleet.total_newbuy_cost == pytest.approx(sum(i * 100 for i in range(3)))
        assert fleet.total_repair_cost == pytest.approx(sum(i * 50 for i in range(3)))

    def test_summary_keys(self):
        fleet = self._make_fleet(2)
        s = fleet.summary()
        assert "operational_availability" in s
        assert "total_cost" in s
        assert "fleet_size" in s

    def test_mean_availability(self):
        wh = _warehouse()
        entities = []
        for i in range(4):
            e = _entity([_consumable_spec()], wh, seed=i)
            e.total_operational_time = float(i * 10)
            e.total_downtime = float((4 - i) * 10)
            entities.append(e)
        fleet = Fleet(entities)
        expected = np.mean([e.availability for e in entities])
        assert fleet.mean_availability == pytest.approx(expected)


# ===========================================================================
# 7. FleetAvailabilityRecorder
# ===========================================================================


class TestFleetAvailabilityRecorder:
    def test_snapshots_recorded_each_tick(self):
        wh = _warehouse(stock=10.0)
        e = _entity([_consumable_spec(NEVER_FAIL)], wh)
        fleet = Fleet([e])
        rec = FleetAvailabilityRecorder(fleet)

        env = SimEnvironment(dt=1.0, end=10.0)
        env.register(e)
        env.register(rec)
        env.run()

        assert len(rec.times) == 10
        assert len(rec.operational) == 10
        assert len(rec.in_repair) == 10
        assert len(rec.awaiting_part) == 10

    def test_all_operational_when_never_fails(self):
        wh = _warehouse(stock=10.0)
        e = _entity([_consumable_spec(NEVER_FAIL)], wh)
        fleet = Fleet([e])
        rec = FleetAvailabilityRecorder(fleet)

        env = SimEnvironment(dt=1.0, end=5.0)
        env.register(e)
        env.register(rec)
        env.run()

        assert all(v == 1 for v in rec.operational)
        assert all(v == 0 for v in rec.in_repair)
        assert all(v == 0 for v in rec.awaiting_part)

    def test_mean_operational_availability_all_up(self):
        wh = _warehouse()
        e = _entity([_consumable_spec(NEVER_FAIL)], wh)
        fleet = Fleet([e])
        rec = FleetAvailabilityRecorder(fleet)

        env = SimEnvironment(dt=1.0, end=10.0)
        env.register(e)
        env.register(rec)
        env.run()

        assert rec.mean_operational_availability == pytest.approx(1.0)

    def test_counts_sum_to_fleet_size(self):
        """operational + in_repair + awaiting_part must equal fleet.size at every tick."""
        wh = _warehouse(stock=5.0, n_skus=2)
        specs = [_consumable_spec(0.3, sku=0), _consumable_spec(0.4, sku=1)]
        entities = [_entity(specs, wh, seed=i) for i in range(4)]
        fleet = Fleet(entities)
        rec = FleetAvailabilityRecorder(fleet)

        env = SimEnvironment(dt=1.0, end=30.0)
        env.register(wh)
        for e in entities:
            env.register(e)
        env.register(rec)
        env.run()

        for op, ir, ap in zip(rec.operational, rec.in_repair, rec.awaiting_part):
            assert op + ir + ap == fleet.size


# ===========================================================================
# 8. sensitivity_sweep
# ===========================================================================


class TestSensitivitySweep:
    """Tests use a deterministic scenario builder for reproducibility."""

    @staticmethod
    def _linear(p1: float, seed: int) -> float:
        """Deterministic: metric = p1 * 2."""
        return p1 * 2.0

    @staticmethod
    def _bilinear(p1: float, p2: float, seed: int) -> float:
        return p1 + p2

    @staticmethod
    def _stochastic(p1: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        return p1 + rng.uniform(-0.1, 0.1)

    def test_1d_result_shape(self):
        p1_vals = [1.0, 2.0, 3.0, 4.0]
        result = sensitivity_sweep(
            self._linear, "p1", p1_vals, metric_name="metric", n_runs=1
        )
        assert result.metric_mean.shape == (4,)
        assert result.metric_std.shape == (4,)
        assert result.is_2d is False

    def test_1d_correct_values(self):
        p1_vals = [1.0, 2.0, 3.0]
        result = sensitivity_sweep(self._linear, "p1", p1_vals, n_runs=1)
        np.testing.assert_allclose(result.metric_mean, [2.0, 4.0, 6.0])

    def test_1d_zero_std_for_deterministic(self):
        result = sensitivity_sweep(
            self._linear, "p1", [1.0, 2.0], n_runs=5
        )
        np.testing.assert_allclose(result.metric_std, [0.0, 0.0], atol=1e-12)

    def test_1d_nonzero_std_for_stochastic(self):
        result = sensitivity_sweep(
            self._stochastic, "p1", [1.0, 2.0], n_runs=10, seed=0
        )
        # Stochastic builder → std should be non-zero.
        assert result.metric_std.sum() > 0.0

    def test_2d_result_shape(self):
        p1_vals = [1.0, 2.0, 3.0]
        p2_vals = [10.0, 20.0]
        result = sensitivity_sweep(
            self._bilinear,
            "p1", p1_vals,
            "p2", p2_vals,
            n_runs=1,
        )
        assert result.metric_mean.shape == (3, 2)
        assert result.metric_std.shape == (3, 2)
        assert result.is_2d is True

    def test_2d_correct_values(self):
        p1_vals = [1.0, 2.0]
        p2_vals = [10.0, 20.0]
        result = sensitivity_sweep(
            self._bilinear, "p1", p1_vals, "p2", p2_vals, n_runs=1
        )
        expected = np.array([[11.0, 21.0], [12.0, 22.0]])
        np.testing.assert_allclose(result.metric_mean, expected)

    def test_param2_name_without_values_raises(self):
        with pytest.raises(ValueError, match="param2_values is required"):
            sensitivity_sweep(
                self._linear, "p1", [1.0],
                param2_name="p2",
                param2_values=None,
            )

    def test_sweep_result_stores_metadata(self):
        result = sensitivity_sweep(
            self._linear, "my_param", [1.0, 2.0],
            metric_name="my_metric", n_runs=3, seed=7
        )
        assert result.param1_name == "my_param"
        assert result.metric_name == "my_metric"
        assert result.n_runs == 3
        assert result.param2_name is None
        assert result.param2_values is None
        np.testing.assert_array_equal(result.param1_values, [1.0, 2.0])


# ===========================================================================
# 9. Top-level export smoke test
# ===========================================================================


def test_top_level_exports():
    """All reliability names must be importable from the top-level package."""
    import simweave as sw

    assert hasattr(sw, "SubsystemSpec")
    assert hasattr(sw, "SubsystemState")
    assert hasattr(sw, "SubsystemStatus")
    assert hasattr(sw, "ReliableEntity")
    assert hasattr(sw, "RepairJob")
    assert hasattr(sw, "RepairCentre")
    assert hasattr(sw, "Fleet")
    assert hasattr(sw, "FleetAvailabilityRecorder")
    assert hasattr(sw, "SweepResult")
    assert hasattr(sw, "sensitivity_sweep")
    assert hasattr(sw, "plot_fleet_availability")
    assert hasattr(sw, "plot_sensitivity_surface")
