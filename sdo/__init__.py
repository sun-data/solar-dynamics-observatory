"""
Download and analyze SDO observations.
"""

from ._paths import directory_default
from ._caching import memory

from . import aia

__all__ = [
    "directory_default",
    "memory",
    "aia",
]
