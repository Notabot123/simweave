import pytest

from simeng.discrete.resources import Resource, ResourcePool


def test_acquire_and_release():
    pool = ResourcePool(maxlen=3, name="pool")
    r0 = Resource(name="r0")
    r1 = Resource(name="r1")
    pool.deposit(r0)
    pool.deposit(r1)
    assert len(pool) == 2

    acquired = pool.try_acquire()
    assert acquired is r0
    assert acquired.is_busy
    assert len(pool) == 1

    acquired.release()
    assert len(pool) == 2
    assert not acquired.is_busy


def test_try_acquire_returns_none_when_empty():
    pool = ResourcePool(maxlen=2, name="pool")
    assert pool.try_acquire() is None


def test_deposit_rejects_non_resource():
    pool = ResourcePool(maxlen=2, name="pool")
    with pytest.raises(TypeError):
        pool.deposit("not a resource")


def test_deposit_full_pool_raises():
    pool = ResourcePool(maxlen=1, name="pool")
    pool.deposit(Resource(name="r"))
    with pytest.raises(RuntimeError):
        pool.deposit(Resource(name="overflow"))
