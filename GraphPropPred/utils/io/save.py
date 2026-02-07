"""
Serialize/deserialize by file extension; create_if_not_exist for dirs.
_resolve maps extension to (dump, load) from _save_helpers.IO_HELPERS.
"""
import warnings
from pathlib import Path
from typing import Union, Any
from ._save_helpers import IO_HELPERS
import os


def _resolve(path: str):
    """Return (dump_fn, load_fn) for the path's extension (json, pkl, pickle)."""
    try:
        _, ext = path.split('.')
    except ValueError:
        ext = path.split('.')
        if len(ext) > 2:
            ext = ext[-1]
        else:
            warnings.warn("File extension is missing, the object will be dumped as a binary file. \
            If this is the intended behaviour, make sure tu specify .pkl as file extension.")
            return IO_HELPERS['pkl']

    if ext not in IO_HELPERS:
        raise ValueError(f"unable to figure out extension '{ext}'")

    return IO_HELPERS[ext]


def dump(obj: Any, path: Union[Path, str]):
    """Serialize obj to path; format inferred from extension. Creates parent dirs."""
    path = Path(path)

    path.parent.mkdir(parents=True,
                      exist_ok=True)

    dump_, _ = _resolve(path.name)
    dump_(obj, path)


def load(path: Union[Path, str]) -> Any:
    """Deserialize from path; format inferred from extension."""
    path = Path(path)

    _, load_ = _resolve(path.name)
    return load_(path)


def create_if_not_exist(path: Union[Path, str]):
    """Create directory path (and parents) if they do not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True,
                      exist_ok=True)
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path):
        raise RuntimeError(f'unable to create {path}')