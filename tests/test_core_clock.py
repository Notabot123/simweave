import pytest

from simeng.core.clock import Clock


def test_clock_defaults():
    c = Clock()
    assert c.t == 0.0
    assert c.dt == 1.0


def test_clock_advance_and_reset():
    c = Clock(start=0.0, dt=0.5, end=2.0)
    for _ in range(3):
        c.advance()
    assert c.t == pytest.approx(1.5)
    c.advance()
    assert c.is_finished()
    c.reset()
    assert c.t == 0.0
    assert not c.is_finished()


def test_clock_jump_forward_only():
    c = Clock(start=0.0, dt=1.0)
    c.jump_to(5.0)
    assert c.t == 5.0
    with pytest.raises(ValueError):
        c.jump_to(1.0)


def test_clock_rejects_bad_params():
    with pytest.raises(ValueError):
        Clock(dt=0)
    with pytest.raises(ValueError):
        Clock(start=1, end=1)
    with pytest.raises(ValueError):
        Clock(start=5, end=2)
