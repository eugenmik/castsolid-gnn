# Node-ordering recovery (Phase 0, Task 0)

Raw `snapshots_nodal`/`temps` are in dolfinx DOF order, scrambled vs gmsh node
order. These scripts reconstruct the per-mesh DOF->gmsh
permutation via dolfinx and validate physically. Run in the `fenicsx` env.

Paths default to the repository's `data/` and `runs/` directories and can be
overridden with the `RECOVERY_*` environment variables defined at the top of
each script.

1. `compute_perms.py` — all stems, coord-hash match -> data/features/perms/<stem>.npy
   + runs/recovery_report.csv
2. `reprocess_failures.py` — re-do `no_bijection` stems with KDTree nearest match.

Result: 9710/9710 perms, all physically valid. build_dataset applies the perm.
