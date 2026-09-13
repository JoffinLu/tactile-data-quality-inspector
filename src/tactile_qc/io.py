"""RCT tactile data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def load_rct(path: PathLike):
    """Load an RCT tactile data file and return a structured representation.

    Parameters
    ----------
    path : str | Path
        Path to the RCT data file.

    Returns
    -------
    TactileData
        Parsed tactile data. Concrete format to be defined in stage 2.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    NotImplementedError
        Until the stage 2 loader lands.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RCT data file not found: {path}")
    raise NotImplementedError("RCT loading is implemented in stage 2")
