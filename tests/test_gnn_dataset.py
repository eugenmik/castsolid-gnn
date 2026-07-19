import numpy as np
from torch_geometric.loader import DataLoader
from src.gnn_dataset import SolidGraphDataset


def _cache(tmp_path):
    g = tmp_path / "graphs"; g.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ["A", "B"]:
        N = 40
        np.savez(g / f"{name}.npz", X=rng.normal(size=(N, 9)).astype(np.float32),
                 y=np.abs(rng.normal(size=N)).astype(np.float32) * 10,
                 pos=rng.normal(size=(N, 3)).astype(np.float32) * 20)
    np.savez(tmp_path / "norm_stats.npz",
             feat_mean=np.zeros(9), feat_std=np.ones(9),
             y_log_mean=np.float64(0.0), y_log_std=np.float64(1.0))
    return tmp_path


def test_dataset_builds_graphs(tmp_path):
    root = _cache(tmp_path)
    ds = SolidGraphDataset(str(root), ["A", "B"], k=8)
    d = ds[0]
    assert d.x.shape == (40, 9)
    assert d.edge_index.shape[0] == 2 and d.edge_index.shape[1] > 0
    assert d.edge_attr.shape[1] == 4
    assert d.y.shape == (40,)
    assert d.pos.shape == (40, 3)


def test_dataloader_batches(tmp_path):
    root = _cache(tmp_path)
    ds = SolidGraphDataset(str(root), ["A", "B"], k=8)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch.x.shape[0] == 80
    assert int(batch.batch.max()) == 1


def test_augment_changes_edge_attr_not_features(tmp_path):
    root = _cache(tmp_path)
    d0 = SolidGraphDataset(str(root), ["A"], k=8, augment=False)[0]
    d1 = SolidGraphDataset(str(root), ["A"], k=8, augment=True)[0]
    assert not np.allclose(d0.edge_attr.numpy(), d1.edge_attr.numpy())   # rotated
    np.testing.assert_allclose(d0.x.numpy(), d1.x.numpy())               # scalars unchanged
