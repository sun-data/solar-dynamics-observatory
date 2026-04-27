"""
Download and prepare observations from the Atmospheric Imaging Assembly (AIA)
"""

from ._filtergrams import Filtergram
from ._aia import open

__all__ = [
    "Filtergram",
    "open",
]
