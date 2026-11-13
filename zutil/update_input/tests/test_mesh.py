"""Reading boundary zone ids from a zCFD native h5 mesh.

The native mesh keeps one row per face in the ``mesh`` group: ``faceBC`` is
the boundary type code the mesh converter wrote, which a v1 BC ``ref``
matched directly (wall 3, inflow 4, outflow 5, symmetry 7, farfield 9), and
column 0 of ``faceInfo`` is the face's zone id. Interior faces carry code 0
or 2 and are left out.
"""

from __future__ import annotations

import pytest

from zutil.update_input.mesh import MeshReadError, read_mesh

h5py = pytest.importorskip("h5py", reason="h5py is needed to write a test mesh")
np = pytest.importorskip("numpy")


def write_mesh(path, face_bc, zones, flat_info=False):
    info = np.stack([np.asarray(zones), np.zeros(len(zones), dtype=int)], axis=1)
    with h5py.File(path, "w") as f:
        g = f.create_group("mesh")
        g.attrs["numFaces"] = np.array([len(face_bc)])
        g.create_dataset("faceBC", data=np.asarray(face_bc, dtype="i4"))
        g.create_dataset("faceInfo", data=info.ravel() if flat_info else info)


def test_zone_ids_by_type_code(tmp_path):
    path = tmp_path / "wing.h5"
    write_mesh(
        path,
        face_bc=[0, 0, 2, 3, 3, 3, 9, 9, 7, 3],
        zones=[0, 0, 0, 5, 5, 6, 12, 12, 14, 4],
    )
    mesh = read_mesh(path)
    assert mesh.zones_by_type == {3: [4, 5, 6], 7: [14], 9: [12]}


def test_label_is_the_mesh_stem(tmp_path):
    path = tmp_path / "wing.h5"
    write_mesh(path, face_bc=[3], zones=[1])
    mesh = read_mesh(path)
    assert mesh.label == "wing"
    assert mesh.path == str(path)


def test_flat_face_info_is_read_as_pairs(tmp_path):
    path = tmp_path / "flat.h5"
    write_mesh(path, face_bc=[3, 4], zones=[7, 8], flat_info=True)
    assert read_mesh(path).zones_by_type == {3: [7], 4: [8]}


def test_missing_dataset_is_named(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f.create_group("mesh").create_dataset("faceBC", data=np.array([3]))
    with pytest.raises(MeshReadError, match="faceInfo"):
        read_mesh(path)


def test_not_an_h5_file(tmp_path):
    path = tmp_path / "wing.h5"
    path.write_text("not a mesh")
    with pytest.raises(MeshReadError, match="wing.h5"):
        read_mesh(path)


def test_missing_file(tmp_path):
    with pytest.raises(MeshReadError, match="does not exist"):
        read_mesh(tmp_path / "nowhere.h5")


def test_cell_zones_are_the_distinct_cell_zone_ids(tmp_path):
    path = tmp_path / "wheel.h5"
    write_mesh(path, face_bc=[3], zones=[1])
    with h5py.File(path, "a") as f:
        f["mesh"].create_dataset("cellZone", data=np.array([0, 17, 0, 17, 4]))
    assert read_mesh(path).cell_zones == [0, 4, 17]


def test_a_mesh_without_cell_zones_is_all_zone_0(tmp_path):
    path = tmp_path / "rotor.h5"
    write_mesh(path, face_bc=[3], zones=[1])
    assert read_mesh(path).cell_zones == [0]
