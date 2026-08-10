"""Small public helper shared by the density-targeted meshing pipeline."""

from __future__ import annotations


def search_elem_size(build_fn, out_msh: str, target_nodes: int,
                     max_iter: int = 3) -> tuple[int, float]:
    """Refine element size assuming node count scales approximately as h^-3."""
    n_nodes, elem_size = build_fn(out_msh, None)
    for _ in range(max_iter - 1):
        if n_nodes <= 0:
            break
        elem_size = elem_size * (n_nodes / target_nodes) ** (1.0 / 3.0)
        n_nodes, elem_size = build_fn(out_msh, elem_size)
    return n_nodes, elem_size


_search_elem_size = search_elem_size
