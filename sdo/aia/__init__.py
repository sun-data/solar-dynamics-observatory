"""
Download and prepare observations from the Atmospheric Imaging Assembly (AIA)
"""

from ._data import (
    urls_jsoc,
    download,
    prep,
)
from ._filtergrams import Filtergram
from ._aia import open

__all__ = [
    "urls_jsoc",
    "download",
    "prep",
    "Filtergram",
    "open",
]
