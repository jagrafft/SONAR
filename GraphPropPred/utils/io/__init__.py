"""
I/O helpers for GraphPropPred: dump, load, create_if_not_exist, join.
Serialization by file extension (json, pkl) via save._resolve.
"""
from .save import dump, load, create_if_not_exist
import os


def join(*paths):
    """Join path components with os.path.join."""
    return os.path.join(*paths)
