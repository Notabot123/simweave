"""Discrete-event primitives: queues, services, resources, arrival generators."""

from simweave.discrete.properties import (
    EntityProperties,
    exponential,
    uniform,
    normal,
    deterministic,
    set_default_seed,
)
from simweave.discrete.queues import Queue, PriorityQueue
from simweave.discrete.resources import Resource, ResourcePool
from simweave.discrete.services import Service, ArrivalGenerator

__all__ = [
    "EntityProperties",
    "exponential",
    "uniform",
    "normal",
    "deterministic",
    "set_default_seed",
    "Queue",
    "PriorityQueue",
    "Resource",
    "ResourcePool",
    "Service",
    "ArrivalGenerator",
]
