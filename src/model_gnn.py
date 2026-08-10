"""MeshGraphNet-style encode-process-decode GNN predicting standardized
log1p(solidification_time) per node; global scalar = max over nodes."""
import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def zero_init_input_columns(model, n_new):
    """Zero the trailing n_new weight columns of the first Linear layer in node_encoder.

    This is used when extending the input dim (e.g., 13 → 15 for new BC channels)
    to ensure new columns start at zero and contribute nothing at initialization.

    Args:
        model: SolidGNN instance
        n_new: Number of trailing columns to zero in the first Linear weight matrix
    """
    with torch.no_grad():
        model.node_encoder[0][0].weight[:, -n_new:] = 0


class ProcessorBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.edge_mlp = _mlp(3 * h, h, h)
        self.edge_norm = nn.LayerNorm(h)
        self.node_mlp = _mlp(2 * h, h, h)
        self.node_norm = nn.LayerNorm(h)

    def forward(self, hnode, edge_index, e):
        src, dst = edge_index
        e_new = self.edge_norm(self.edge_mlp(torch.cat([hnode[src], hnode[dst], e], dim=-1)) + e)
        agg = scatter(e_new, dst, dim=0, dim_size=hnode.shape[0], reduce="mean")
        h_new = self.node_norm(self.node_mlp(torch.cat([hnode, agg], dim=-1)) + hnode)
        return h_new, e_new


class FiLM(nn.Module):
    """g -> (gamma, beta) per hidden channel; zero-init => gamma=0,beta=0 => identity."""
    def __init__(self, g_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(g_dim, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, g_nodes):
        gamma, beta = self.net(g_nodes).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class SolidGNN(nn.Module):
    def __init__(self, node_in=9, edge_in=4, hidden=128, n_blocks=12, two_head=False,
                 film=False, g_dim=5, deriv_heads=False):
        super().__init__()
        self.node_encoder = nn.Sequential(_mlp(node_in, hidden, hidden), nn.LayerNorm(hidden))
        self.edge_encoder = nn.Sequential(_mlp(edge_in, hidden, hidden), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList([ProcessorBlock(hidden) for _ in range(n_blocks)])
        self.decoder = _mlp(hidden, hidden // 2, 1)
        self.two_head = two_head
        if two_head:
            self.scale_head = _mlp(hidden, hidden // 2, 1)
        self.film = film
        self.films = nn.ModuleList([FiLM(g_dim, hidden) for _ in range(n_blocks)]) if film else None
        self.deriv_heads = deriv_heads
        if deriv_heads:
            # Optional gradT/dTdt output heads. They hang off the SAME final
            # hidden state h that the decoder consumes (after the FiLM/blocks
            # loop, before the two_head mean/scale recomposition) -- they read
            # h but never modify it, so the frozen y_nodal/y_global path below
            # is unaffected by their presence.
            self.grad_head = _mlp(hidden, hidden // 2, 1)
            self.dtdt_head = _mlp(hidden, hidden // 2, 1)

    def forward(self, x, edge_index, edge_attr, batch, g=None):
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        g_nodes = g[batch] if (self.film and g is not None) else None
        for i, blk in enumerate(self.blocks):
            h, e = blk(h, edge_index, e)
            if g_nodes is not None:
                h = self.films[i](h, g_nodes)
        if self.deriv_heads:
            deriv = torch.cat([self.grad_head(h), self.dtdt_head(h)], dim=-1)
        y_nodal = self.decoder(h).squeeze(-1)
        if self.two_head:
            dev = y_nodal - scatter(y_nodal, batch, dim=0, reduce="mean")[batch]
            scale = self.scale_head(scatter(h, batch, dim=0, reduce="mean")).squeeze(-1)
            y_nodal = dev + scale[batch]
        y_global = scatter(y_nodal, batch, dim=0, reduce="max")
        if self.deriv_heads:
            return y_nodal, y_global, deriv
        return y_nodal, y_global


def attach_deriv_heads(model, state_dict):
    """Load a (possibly headless) state_dict into a `deriv_heads=True` model
    via `load_state_dict(strict=False)`, asserting the ONLY missing keys are
    the two new heads' parameters (`grad_head.*` / `dtdt_head.*`).

    This is how a production checkpoint trained before the deriv-heads
    feature existed gets loaded into a `deriv_heads=True` model: everything
    else must match exactly, or this raises loudly instead of silently
    leaving some backbone tensor at its random init.

    Args:
        model: a `SolidGNN(deriv_heads=True)` instance.
        state_dict: source state_dict (e.g. from an old checkpoint's `model`
            entry). May or may not already contain grad_head/dtdt_head keys.

    Returns:
        list of missing key names (all `grad_head.*`/`dtdt_head.*`).

    Raises:
        ValueError: if any non-head (backbone) key is missing after the load,
            or if the state_dict contains keys the model doesn't recognize.
    """
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise ValueError(
            f"attach_deriv_heads: unexpected extra keys in state_dict "
            f"(not present in model): {list(result.unexpected_keys)}"
        )
    non_head_missing = [
        k for k in result.missing_keys
        if not k.startswith("grad_head.") and not k.startswith("dtdt_head.")
    ]
    if non_head_missing:
        raise ValueError(
            f"attach_deriv_heads: missing backbone keys after load (expected "
            f"only grad_head.*/dtdt_head.* to be missing): {non_head_missing}"
        )
    return list(result.missing_keys)
