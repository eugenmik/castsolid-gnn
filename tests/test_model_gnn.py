import pytest
import torch
from src.model_gnn import SolidGNN, attach_deriv_heads

# Frozen regression guard: the exact state_dict key set for a
# SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True) model
# BEFORE the deriv_heads feature was added. deriv_heads=False must reproduce
# this set exactly (no grad_head/dtdt_head keys, no renamed backbone keys).
_PRE_DERIV_HEADS_KEYS = {
    "blocks.0.edge_mlp.0.bias", "blocks.0.edge_mlp.0.weight",
    "blocks.0.edge_mlp.2.bias", "blocks.0.edge_mlp.2.weight",
    "blocks.0.edge_norm.bias", "blocks.0.edge_norm.weight",
    "blocks.0.node_mlp.0.bias", "blocks.0.node_mlp.0.weight",
    "blocks.0.node_mlp.2.bias", "blocks.0.node_mlp.2.weight",
    "blocks.0.node_norm.bias", "blocks.0.node_norm.weight",
    "blocks.1.edge_mlp.0.bias", "blocks.1.edge_mlp.0.weight",
    "blocks.1.edge_mlp.2.bias", "blocks.1.edge_mlp.2.weight",
    "blocks.1.edge_norm.bias", "blocks.1.edge_norm.weight",
    "blocks.1.node_mlp.0.bias", "blocks.1.node_mlp.0.weight",
    "blocks.1.node_mlp.2.bias", "blocks.1.node_mlp.2.weight",
    "blocks.1.node_norm.bias", "blocks.1.node_norm.weight",
    "decoder.0.bias", "decoder.0.weight",
    "decoder.2.bias", "decoder.2.weight",
    "edge_encoder.0.0.bias", "edge_encoder.0.0.weight",
    "edge_encoder.0.2.bias", "edge_encoder.0.2.weight",
    "edge_encoder.1.bias", "edge_encoder.1.weight",
    "node_encoder.0.0.bias", "node_encoder.0.0.weight",
    "node_encoder.0.2.bias", "node_encoder.0.2.weight",
    "node_encoder.1.bias", "node_encoder.1.weight",
    "scale_head.0.bias", "scale_head.0.weight",
    "scale_head.2.bias", "scale_head.2.weight",
}


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


def test_two_head_forward_backward_and_shape_decomposition():
    import torch
    from torch_geometric.utils import scatter
    from src.model_gnn import SolidGNN
    m = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True)
    x = torch.randn(12, 9)
    ei = torch.stack([torch.arange(11), torch.arange(1, 12)])
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    ea = torch.randn(ei.shape[1], 4)
    batch = torch.tensor([0] * 6 + [1] * 6)
    yn, yg = m(x, ei, ea, batch)
    assert yn.shape == (12,) and yg.shape == (2,)
    yn.sum().backward()                                    # grads flow to both heads
    assert m.scale_head[0].weight.grad is not None
    # per-graph mean of the output equals the scale head's prediction:
    # dev part is exactly zero-mean per graph by construction
    pm = scatter(yn, batch, dim=0, reduce="mean")
    with torch.no_grad():
        h = m.node_encoder(x); e = m.edge_encoder(ea)
        for blk in m.blocks:
            h, e = blk(h, ei, e)
        scale = m.scale_head(scatter(h, batch, dim=0, reduce="mean")).squeeze(-1)
    torch.testing.assert_close(pm, scale, atol=1e-5, rtol=1e-5)


def test_deriv_heads_false_is_2tuple_and_state_dict_keys_unchanged():
    """deriv_heads=False (the default) must be BIT-identical to the
    pre-deriv_heads module: same 2-tuple return, same state_dict key set
    (no grad_head/dtdt_head keys, no accidental backbone renames)."""
    torch.manual_seed(0)
    m = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True)
    N, E, B = 12, 60, 2
    x = torch.randn(N, 9)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 4)
    batch = torch.tensor([0] * 6 + [1] * 6)

    out = m(x, edge_index, edge_attr, batch)
    assert len(out) == 2
    yn, yg = out
    assert yn.shape == (N,)
    assert yg.shape == (B,)

    assert set(m.state_dict().keys()) == _PRE_DERIV_HEADS_KEYS


def test_deriv_heads_true_returns_3tuple_with_correct_deriv_shape():
    torch.manual_seed(0)
    m = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True,
                 deriv_heads=True)
    N, E, B = 12, 60, 2
    x = torch.randn(N, 9)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 4)
    batch = torch.tensor([0] * 6 + [1] * 6)

    out = m(x, edge_index, edge_attr, batch)
    assert len(out) == 3
    yn, yg, deriv = out
    assert yn.shape == (N,)
    assert yg.shape == (B,)
    assert deriv.shape == (N, 2)

    # new heads exist as real modules with grad-tracked params
    deriv.sum().backward()
    assert m.grad_head[0].weight.grad is not None
    assert m.dtdt_head[0].weight.grad is not None


def test_deriv_heads_true_frozen_path_is_bit_identical_to_false():
    """The deriv heads read h but must never alter y_nodal/y_global: a
    deriv_heads=True model loaded from a deriv_heads=False model's state_dict
    (via attach_deriv_heads) must produce torch.equal y_nodal/y_global on the
    same inputs -- true bit-identity of the frozen path, not just close."""
    torch.manual_seed(7)
    base = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True)
    base.eval()

    m = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True,
                 deriv_heads=True)
    attach_deriv_heads(m, base.state_dict())
    m.eval()

    N, E, B = 14, 70, 2
    x = torch.randn(N, 9)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 4)
    batch = torch.tensor([0] * 7 + [1] * 7)

    with torch.no_grad():
        yn_base, yg_base = base(x, edge_index, edge_attr, batch)
        yn_new, yg_new, deriv = m(x, edge_index, edge_attr, batch)

    assert torch.equal(yn_base, yn_new)
    assert torch.equal(yg_base, yg_new)
    assert deriv.shape == (N, 2)


def test_attach_deriv_heads_missing_keys_are_exactly_the_new_heads():
    torch.manual_seed(3)
    base = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True)
    m = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True,
                 deriv_heads=True)

    missing = attach_deriv_heads(m, base.state_dict())

    # Derive the expected set from _mlp's own structure (grad_head/dtdt_head
    # are each an nn.Sequential(Linear, ReLU, Linear) -- 4 param tensors
    # apiece: {0,2}.{weight,bias}), rather than hardcoding a magic count.
    expected = {k for k in m.state_dict() if k.startswith(("grad_head.", "dtdt_head."))}
    assert expected, "expected grad_head/dtdt_head keys to exist on a deriv_heads=True model"
    assert set(missing) == expected

    # every remaining (backbone) key must have been loaded, i.e. not missing
    non_head_missing = [k for k in missing if k not in expected]
    assert non_head_missing == []


def test_attach_deriv_heads_missing_backbone_key_raises_loudly():
    torch.manual_seed(3)
    base = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True)
    m = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2, two_head=True,
                 deriv_heads=True)

    sd = dict(base.state_dict())
    del sd["decoder.0.weight"]  # simulate a corrupted/mismatched checkpoint

    with pytest.raises(ValueError):
        attach_deriv_heads(m, sd)
