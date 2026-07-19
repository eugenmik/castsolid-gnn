import numpy as np
import torch
from src.build_dataset import process_one
from tests.conftest import MINI_MSH


def test_process_one_returns_pos(tmp_path):
    (tmp_path / "pt").mkdir(parents=True)
    (tmp_path / "msh_none" / "boxcor").mkdir(parents=True)
    stem = "BC_10_10_10"
    (tmp_path / "msh_none" / "boxcor" / f"{stem}.msh").write_text(MINI_MSH)
    N = 5
    temps = torch.full((2, N), 1200.0); temps[1, :] = 1100.0
    torch.save({"pos": torch.zeros(N, 3), "edge_index": torch.zeros(2, 1, dtype=torch.long),
                "temps": temps, "times": torch.tensor([0.0, 10.0])},
               tmp_path / "pt" / f"{stem}.pt")
    (tmp_path / "features" / "perms").mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / "features" / "perms" / f"{stem}.npy", np.arange(N, dtype=np.int32))

    rec = process_one(stem, str(tmp_path))
    assert rec is not None
    assert rec["pos"].shape == (5, 3)
    assert rec["X"].shape == (5, 9)
