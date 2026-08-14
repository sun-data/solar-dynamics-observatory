"""
Download and prepare observations from the Helioseismic and Magnetic Imager (HMI)
"""

from ._data import search, urls
from ._magnetograms import Magnetogram
from ._hmi import open

__all__ = [
    "search",
    "urls",
    "Magnetogram",
    "open",
]
