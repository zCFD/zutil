"""Read the boundary zone ids of each type, and the cell zone ids, from a zCFD native h5 mesh.

The ``mesh`` group holds one row per face. ``faceBC`` is the boundary type
code, the value a v1 BC ``ref`` was compared with, and column 0 of
``faceInfo`` is the face's zone id. The optional ``cellZone`` dataset holds
each cell's zone id; a mesh without it has every cell in zone 0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from .model import MeshInfo

INTERIOR_CODES = (0, 2)


class MeshReadError(ValueError):
    """The ``--mesh`` file cannot be read as a zCFD native mesh."""


def read_mesh(path: Union[str, os.PathLike]) -> MeshInfo:
    """The zone ids for each boundary type code, and the cell zone ids, of the mesh at ``path``."""
    path = Path(path)
    if not path.is_file():
        raise MeshReadError(f"the mesh file {path} does not exist")
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        raise MeshReadError(
            f"reading {path.name} needs h5py; install it with 'pip install h5py'"
        ) from e

    try:
        with h5py.File(path, "r") as f:
            if "mesh" not in f:
                raise MeshReadError(
                    f"{path.name} has no 'mesh' group, so it is not a zCFD mesh"
                )
            group = f["mesh"]
            for name in ("faceBC", "faceInfo"):
                if name not in group:
                    raise MeshReadError(
                        f"{path.name} has no 'mesh > {name}' dataset, so the zones of each "
                        "boundary type cannot be read"
                    )
            face_bc = np.asarray(group["faceBC"][()]).astype("i8").ravel()
            face_info = np.asarray(group["faceInfo"][()]).astype("i8")
            cell_zones = (
                np.unique(group["cellZone"][()]).astype(int).tolist()
                if "cellZone" in group
                else [0]
            )
    except OSError as e:
        raise MeshReadError(f"{path.name} cannot be read as an h5 mesh: {e}") from e

    zones = face_info.reshape(len(face_bc), -1)[:, 0]
    boundary = ~np.isin(face_bc, INTERIOR_CODES)
    pairs = np.unique(np.stack([face_bc[boundary], zones[boundary]], axis=1), axis=0)

    zones_by_type: dict[int, list[int]] = {}
    for code, zone in pairs.tolist():
        zones_by_type.setdefault(int(code), []).append(int(zone))
    return MeshInfo(str(path), path.stem, zones_by_type, cell_zones)
