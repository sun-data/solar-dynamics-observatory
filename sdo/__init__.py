"""
Download and analyze SDO observations.
"""

from ._paths import directory_default

from . import jsoc
from . import aia

__all__ = [
    "directory_default",
    "jsoc",
    "aia",
]
