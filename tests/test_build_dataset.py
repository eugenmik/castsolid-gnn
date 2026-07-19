import numpy as np
import torch
from pathlib import Path
from src.build_dataset import find_msh, process_one
from tests.conftest import MINI_MSH


def _make_fake_data(root: Path):
    """Create a data/ tree with one .pt and its paired .msh (5 nodes)."""
    (root / "pt").mkdir(parents=True)
    (root / "msh_none" / "boxcor").mkdir(parents=True)
    stem = "BC_10_10_10"
    (root / "msh_none" / "boxcor" / f"{stem}.msh").write_text(MINI_MSH)
    N = 5
    temps = torch.full((2, N), 1200.0)
    temps[1, :] = 1100.0
    torch.save(
        {"pos": torch.zeros(N, 3), "edge_index": torch.zeros(2, 1, dtype=torch.long),
         "temps": temps, "times": torch.tensor([0.0, 10.0])},
        root / "pt" / f"{stem}.pt",
    )
    # identity recovery permutation (fake data is already node-consistent)
    (root / "features" / "perms").mkdir(parents=True, exist_ok=True)
    np.save(root / "features" / "perms" / f"{stem}.npy", np.arange(N, dtype=np.int32))
    return stem


def test_find_and_process(tmp_path):
    stem = _make_fake_data(tmp_path)
    assert find_msh(stem, str(tmp_path)).endswith(f"{stem}.msh")

    rec = process_one(stem, str(tmp_path))
    assert rec is not None
    assert rec["n_nodes"] == 5
    assert rec["prefix"] == "BC"
    assert rec["n_adiab"] == 0
    assert rec["is_fully_cooled"] is True
    assert rec["X"].shape == (5, 9)
    np.testing.assert_allclose(rec["y"], 8.0, rtol=1e-5)


def test_perm_is_applied(tmp_path):
    """A non-identity perm must reorder temps into gmsh node order before the
    target is computed. temps_gmsh[:, g] = temps_dof."""
    (tmp_path / "pt").mkdir(parents=True)
    (tmp_path / "msh_none" / "boxcor").mkdir(parents=True)
    stem = "BC_10_10_10"
    (tmp_path / "msh_none" / "boxcor" / f"{stem}.msh").write_text(MINI_MSH)
    N = 5
    # DOF-order temps: column 0 crosses 1120 at t=8; columns 1..4 already solid (y=0)
    temps = torch.full((2, N), 1000.0); temps[1, :] = 900.0   # all already below -> y=0
    temps[0, 0] = 1200.0; temps[1, 0] = 1100.0                # column 0 -> y=8
    torch.save({"pos": torch.zeros(N, 3), "edge_index": torch.zeros(2, 1, dtype=torch.long),
                "temps": temps, "times": torch.tensor([0.0, 10.0])},
               tmp_path / "pt" / f"{stem}.pt")
    g = np.array([2, 0, 1, 3, 4], dtype=np.int32)   # DOF col 0 -> gmsh node 2
    (tmp_path / "features" / "perms").mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / "features" / "perms" / f"{stem}.npy", g)

    rec = process_one(stem, str(tmp_path))
    assert rec is not None
    assert int(np.argmax(rec["y"])) == 2          # the 8s value moved to gmsh node 2
    np.testing.assert_allclose(rec["y"][2], 8.0, rtol=1e-5)


def test_no_perm_returns_none(tmp_path):
    """A stem with a mesh but no recovery perm is excluded (returns None)."""
    (tmp_path / "pt").mkdir(parents=True)
    (tmp_path / "msh_none" / "boxcor").mkdir(parents=True)
    stem = "BC_10_10_10"
    (tmp_path / "msh_none" / "boxcor" / f"{stem}.msh").write_text(MINI_MSH)
    N = 5
    temps = torch.full((2, N), 1200.0); temps[1, :] = 1100.0
    torch.save({"pos": torch.zeros(N, 3), "edge_index": torch.zeros(2, 1, dtype=torch.long),
                "temps": temps, "times": torch.tensor([0.0, 10.0])},
               tmp_path / "pt" / f"{stem}.pt")
    assert process_one(stem, str(tmp_path)) is None   # no perm file -> excluded


def test_missing_msh_returns_none(tmp_path):
    (tmp_path / "pt").mkdir(parents=True)
    torch.save({"pos": torch.zeros(1, 3), "edge_index": torch.zeros(2, 1, dtype=torch.long),
                "temps": torch.full((2, 1), 1200.0), "times": torch.tensor([0.0, 1.0])},
               tmp_path / "pt" / "ZZ_01.pt")
    assert process_one("ZZ_01", str(tmp_path)) is None
