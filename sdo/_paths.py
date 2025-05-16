import pathlib

__all__ = [
    "directory_default",
]


directory_default = pathlib.Path.home() / ".sdo/cache"
"""The default directory for downloaded images."""
