from pathlib import Path

import pytest

from zutil.validate import in_case_directory


@pytest.fixture(autouse=True)
def _decks_validate_without_their_files():
    """The test decks name field files that are not shipped beside them."""
    with in_case_directory(None):
        yield


@pytest.fixture
def stand_in_mesh():
    """Writes an h5 mesh with no boundary faces and no cellZone dataset.

    The mesh value is then set and nothing else; every cell is in cell zone 0,
    so a frame on zone 0 covers the whole mesh.
    """
    h5py = pytest.importorskip("h5py", reason="h5py is needed to write a stand-in mesh")
    np = pytest.importorskip("numpy")

    def write(path: Path) -> Path:
        with h5py.File(path, "w") as f:
            g = f.create_group("mesh")
            g.create_dataset("faceBC", data=np.array([0]))
            g.create_dataset("faceInfo", data=np.array([[0, 0]]))
        return path

    return write


@pytest.fixture
def run_deck(monkeypatch):
    """Runs converted deck text as ``out_v2.py`` in ``where`` and returns its parameters."""

    def run(text: str, where: Path) -> dict:
        namespace = {"__file__": str(where / "out_v2.py")}
        with monkeypatch.context() as m:
            m.chdir(where)
            exec(compile(text, "out_v2.py", "exec"), namespace)
        return namespace["parameters"]

    return run
