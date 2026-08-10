import torch
from src.model_gnn import SolidGNN, zero_init_input_columns


def test_zero_init_last_columns():
    m = SolidGNN(node_in=15, hidden=16, n_blocks=1)
    zero_init_input_columns(m, n_new=2)
    w = m.node_encoder[0][0].weight  # first Linear of the node_encoder _mlp
    assert torch.count_nonzero(w[:, -2:]) == 0 and torch.count_nonzero(w[:, :13]) > 0
