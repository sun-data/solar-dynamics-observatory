"""
Download and analyze SDO observations.
"""

from ._paths import directory_default
from ._caching import memory
from ._data import download

from . import aia
from . import hmi

__all__ = [
    "directory_default",
    "memory",
    "download",
    "aia",
    "hmi",
]
