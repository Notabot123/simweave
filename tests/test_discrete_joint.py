"""ArrivalGenerator(joint=...), EntityProperties.from_draw and Service integration."""
from __future__ import annotations

import numpy as np
import pytest

import simweave as sw
from simweave.core.entity import Entity
from simweave.stats import Deterministic, Exponential, GaussianCopula, JointSampler, kendall_tau


def _env():
    return sw.SimEnvironment(dt=0.05)


def _comonotone():
    # tau ~ 1 via a near-singular Gaussian: gap and size are (almost) the same quantile.
    return GaussianCopula(0.999999)


def test_legacy_path_unchanged_and_validation():
    env = _env()
    q = sw.Queue(maxlen=10_000)
    g = sw.ArrivalGenerator(sw.exponential(1.0), lambda e: Entity(), q, rng=np.random.default_rng(1))
    assert g.joint is None
    with pytest.raises(ValueError):
        sw.ArrivalGenerator(None, lambda e, d: Entity(), q)
    js = JointSampler(GaussianCopula(0.3), {"interarrival": Exponential(1.0), "x": Exponential(1.0)})
    with pytest.raises(ValueError):
        sw.ArrivalGenerator(sw.exponential(1.0), lambda e, d: Entity(), q, joint=js)
    with pytest.raises(ValueError):
        sw.ArrivalGenerator(None, lambda e, d: Entity(), q, joint=js, gap_alignment="sideways")
    with pytest.raises(ValueError):
        sw.ArrivalGenerator(None, lambda e, d: Entity(), q, joint=js, interarrival_key="nope")


def _run(alignment, seed=3, n_ticks=4000, copula=None):
    env = _env()
    q = sw.Queue(maxlen=100_000)
    js = JointSampler(
        copula or GaussianCopula(0.7),
        {"interarrival": Exponential(1.0), "size": Exponential(1.0)},
    )
    seen = []

    def factory(e, draw):
        ent = Entity()
        ent.draw = draw
        ent.t = e.clock.t
        seen.append(ent)
        return ent

    gen = sw.ArrivalGenerator(None, factory, q, rng=np.random.default_rng(seed), joint=js,
                              gap_alignment=alignment)
    env.register(gen)
    env.run(n_ticks * 0.05)
    return seen


@pytest.mark.parametrize("alignment", ["before", "after"])
def test_attributes_correlate_with_the_right_gap(alignment):
    seen = _run(alignment, n_ticks=40_000, copula=_comonotone())
    t = np.array([e.t for e in seen])
    gaps = np.diff(t)  # observed gap preceding entity i (i >= 1)
    size = np.array([e.draw["size"] for e in seen])
    ours = kendall_tau(gaps[:-1], size[1:-1]) if alignment == "before" else kendall_tau(gaps[1:], size[1:-1])
    # "before": size_i tracks the gap just before it; "after": size_i tracks the gap after it.
    other = kendall_tau(gaps[1:], size[1:-1]) if alignment == "before" else kendall_tau(gaps[:-1], size[1:-1])
    assert ours > 0.8 and abs(other) < 0.1


def test_deterministic_given_seed_and_isolated_stream():
    a = [e.draw["size"] for e in _run("before", seed=5)]
    b = [e.draw["size"] for e in _run("before", seed=5)]
    c = [e.draw["size"] for e in _run("before", seed=6)]
    assert a == b and a != c


def test_from_draw_feeds_service_times():
    draw = {"interarrival": 1.0, "service_time": 0.4, "size": 7.0}
    p = sw.EntityProperties.from_draw(draw, entity_type="job")
    assert p.entity_type == "job" and p.extras == {"interarrival": 1.0, "size": 7.0}
    assert p.draw_service_time(np.random.default_rng(0)) == 0.4
    assert sw.EntityProperties.from_draw(draw, service_key=None).extras["service_time"] == 0.4
    with pytest.raises(KeyError):
        sw.EntityProperties.from_draw(draw, service_key="nope")


def test_service_uses_drawn_service_times_and_littles_law():
    env = sw.SimEnvironment(dt=0.02)
    sink = sw.Service(capacity=1, buffer_size=100_000, rng=np.random.default_rng(1))
    js = JointSampler(
        GaussianCopula(-0.5),
        {"interarrival": Exponential(1.0), "service_time": Exponential(0.6)},
    )
    served = []

    def factory(e, draw):
        ent = Entity()
        ent.sim_properties = sw.EntityProperties.from_draw(draw)
        served.append(draw["service_time"])
        return ent

    gen = sw.ArrivalGenerator(None, factory, sink, rng=np.random.default_rng(2), joint=js)
    env.register(gen)
    env.register(sink)
    env.run(3000.0)
    assert sink.completed_count > 2000
    mean_service = np.mean(served)
    assert mean_service == pytest.approx(0.6, rel=0.1)
    # total busy time equals the sum of the drawn service times of finished jobs (+ in progress)
    assert sum(ch.busy_time for ch in sink.channels) == pytest.approx(
        sum(served[: sink.completed_count]), rel=0.05
    )


def test_serial_process_plugs_into_unchanged_interarrival_slot():
    env = _env()
    q = sw.Queue(maxlen=100_000)
    proc = sw.MarkovCopulaProcess(sw.GumbelCopula.from_kendall_tau(0.7), Exponential(1.0))
    gen = sw.ArrivalGenerator(proc, lambda e: Entity(), q, rng=np.random.default_rng(1))
    env.register(gen)
    env.run(2000.0)
    assert 1500 < gen.generated < 2500


def test_dependence_changes_queue_waits():
    """Negative gap/service dependence (big job, short following gap) raises waits."""

    def mean_wait(copula, seed):
        env = sw.SimEnvironment(dt=0.05)
        svc = sw.Service(capacity=1, buffer_size=100_000, rng=np.random.default_rng(seed))
        js = JointSampler(copula, {"interarrival": Exponential(1.0), "service_time": Exponential(0.8)})
        gen = sw.ArrivalGenerator(
            None, lambda e, d: _job(d), svc, rng=np.random.default_rng(seed), joint=js,
            gap_alignment="after",
        )
        env.register(gen)
        env.register(svc)
        env.run(6000.0)
        return svc.average_residence()

    def _job(d):
        ent = Entity()
        ent.sim_properties = sw.EntityProperties.from_draw(d)
        return ent

    indep = np.mean([mean_wait(GaussianCopula(1e-9), s) for s in range(4)])
    neg = np.mean([mean_wait(GaussianCopula(-0.8), s) for s in range(4)])
    assert neg > 1.15 * indep


def test_monte_carlo_with_processes_executor_is_picklable():
    res = sw.run_monte_carlo(_mc_scenario, n_runs=2, executor="processes", n_workers=2)
    ref = sw.run_monte_carlo(_mc_scenario, n_runs=2)
    assert np.allclose(res.samples, ref.samples)


def _mc_scenario(seed):
    env = sw.SimEnvironment(dt=0.05)
    q = sw.Queue(maxlen=100_000)
    js = JointSampler(GaussianCopula(0.5), {"interarrival": Exponential(1.0), "x": Exponential(1.0)})
    gen = sw.ArrivalGenerator(None, lambda e, d: Entity(), q, rng=np.random.default_rng(seed), joint=js)
    env.register(gen)
    env.run(200.0)
    return gen.generated
