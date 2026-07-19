import numpy as np
import torch
from pathlib import Path
from src.build_dataset import build
from tests.conftest import MINI_MSH


def _fake(root: Path, stem, msh_set, subdir):
    (root / "pt").mkdir(parents=True, exist_ok=True)
    (root / msh_set / subdir).mkdir(parents=True, exist_ok=True)
    (root / msh_set / subdir / f"{stem}.msh").write_text(MINI_MSH)
    N = 5
    temps = torch.full((2, N), 1200.0); temps[1, :] = 1100.0
    torch.save({"pos": torch.zeros(N, 3), "edge_index": torch.zeros(2, 1, dtype=torch.long),
                "temps": temps, "times": torch.tensor([0.0, 10.0])},
               root / "pt" / f"{stem}.pt")
    (root / "features" / "perms").mkdir(parents=True, exist_ok=True)
    np.save(root / "features" / "perms" / f"{stem}.npy", np.arange(N, dtype=np.int32))


def test_build_writes_cache(tmp_path):
    _fake(tmp_path, "BC_10_10_10", "msh_none", "boxcor")
    _fake(tmp_path, "BC_10_10_10_+X", "msh_half", "boxcor")
    out = tmp_path / "features"
    build(data_root=str(tmp_path), out_dir=str(out), n_workers=1)

    assert (out / "graphs" / "BC_10_10_10.npz").exists()
    meta = np.load(out / "meta.npz", allow_pickle=True)
    assert set(meta["stem"]) == {"BC_10_10_10", "BC_10_10_10_+X"}
    assert sorted(meta["n_adiab"].tolist()) == [0, 1]

    ns = np.load(out / "norm_stats.npz")
    assert ns["feat_mean"].shape == (9,)
    assert np.isfinite(ns["y_log_mean"])
