import torch
from src.model_gnn import SolidGNN


def test_forward_shapes():
    torch.manual_seed(0)
    N, E, B = 50, 400, 2
    x = torch.randn(N, 9)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 4)
    batch = torch.zeros(N, dtype=torch.long); batch[25:] = 1
    model = SolidGNN(n_blocks=3, hidden=32)
    yn, yg = model(x, edge_index, edge_attr, batch)
    assert yn.shape == (N,)
    assert yg.shape == (B,)


def test_global_is_max_of_nodal():
    torch.manual_seed(1)
    N, E = 30, 200
    x = torch.randn(N, 9); edge_index = torch.randint(0, N, (2, E)); edge_attr = torch.randn(E, 4)
    batch = torch.zeros(N, dtype=torch.long)
    model = SolidGNN(n_blocks=2, hidden=16).eval()
    with torch.no_grad():
        yn, yg = model(x, edge_index, edge_attr, batch)
    torch.testing.assert_close(yg[0], yn.max())
