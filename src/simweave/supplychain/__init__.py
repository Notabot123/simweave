"""Inventory and supply-chain simulation primitives."""

from simweave.supplychain.inventory import InventoryItems
from simweave.supplychain.warehouse import Warehouse

__all__ = ["InventoryItems", "Warehouse"]

# The optimisation module is imported lazily so that simweave is usable
# without scipy. Access it via ``from simweave.supplychain import optimization``.
