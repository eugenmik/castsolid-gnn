import torch
from src.model_gnn import FiLM, SolidGNN


def test_film_zero_init_is_identity():
    f = FiLM(g_dim=5, hidden=16)
    h = torch.randn(7, 16); g = torch.randn(7, 5)
    assert torch.allclose(f(h, g), h, atol=1e-6)


def test_solidgnn_film_zero_init_matches_no_film():
    """SolidGNN(film=True) with zero-init FiLM params must reproduce
    SolidGNN(film=False) output exactly, given identical base weights,
    identical inputs, a real batch vector, and a real (non-None) g tensor."""
    N, E, B = 20, 120, 3
    x = torch.randn(N, 9)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 4)
    batch = torch.tensor([0] * 7 + [1] * 7 + [2] * 6, dtype=torch.long)
    g = torch.randn(B, 5)

    torch.manual_seed(42)
    base = SolidGNN(n_blocks=3, hidden=16, film=False)

    torch.manual_seed(42)
    filmed = SolidGNN(n_blocks=3, hidden=16, film=True, g_dim=5)
    # base weights (encoder/blocks/decoder) are seeded identically; FiLM layers
    # are zero-init by construction (asserted separately in the test above), so
    # filmed's non-FiLM parameters must match base's exactly.
    for (n1, p1), (n2, p2) in zip(base.named_parameters(), filmed.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)

    base.eval(); filmed.eval()
    with torch.no_grad():
        yn_base, yg_base = base(x, edge_index, edge_attr, batch)
        yn_film, yg_film = filmed(x, edge_index, edge_attr, batch, g=g)

    torch.testing.assert_close(yn_film, yn_base, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(yg_film, yg_base, atol=1e-6, rtol=1e-6)

    # also check g=None on the filmed model is identical (old positional call still works)
    with torch.no_grad():
        yn_film_noG, yg_film_noG = filmed(x, edge_index, edge_attr, batch)
    torch.testing.assert_close(yn_film_noG, yn_base, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(yg_film_noG, yg_base, atol=1e-6, rtol=1e-6)
