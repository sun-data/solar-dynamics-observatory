import joblib
import sdo

__all__ = [
    "path_cache",
    "memory",
]

#: The location on the filesystem where SDO images are stored.
path_cache = sdo.directory_default

#: A representation of the cache which stores intermediate results.
memory = joblib.Memory(location=path_cache, mmap_mode="r", verbose=0)
