"""Pytest fixtures shared across the simeng test suite."""

import numpy as np
import pytest

from simeng.core.entity import Entity
from simeng.discrete.properties import set_default_seed


@pytest.fixture(autouse=True)
def _reset_entity_ids():
    """Reset the global entity ID counter between tests for determinism."""
    Entity.reset_id_counter()
    yield


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def _seed_default_rng():
    set_default_seed(42)
    yield
