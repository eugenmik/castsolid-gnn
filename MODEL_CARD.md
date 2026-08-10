# Model card: solidification surrogate v5.1

## Model details

- **Version:** v5.1
- **Released internally:** July 31, 2026
- **Developer:** Eugen Miknevic, independent foundry engineer
- **Contact:** eugenmiknevic@gmail.com
- **Intended role:** rapid solidification screening

The predictor is a three-model hybrid: a 12-block graph neural network predicts the spatial
field pattern, one gradient-boosted head predicts total time, and another predicts field
spread. A weighted soft-anchor operation recomposes the outputs.

## Intended use

- compare early casting concepts;
- identify candidate last-to-solidify regions;
- triage which designs require a full process simulation;
- support exploratory quoting and design discussions when accompanied by visible limitations.

## Out-of-scope use

- final process sign-off;
- guaranteed defect or porosity prediction;
- mold filling, flow, gas, stress or distortion analysis;
- unattended decisions for geometry outside the training envelope;
- safety-critical or contractual acceptance decisions without independent engineering review.

## Inputs and outputs

Inputs are a millimeter-scale STEP geometry, alloy and mold properties, and optional
multi-body roles for a casting with chills or one sleeve regime. Outputs are per-node
solidification time, total time, a top-decile hot-spot mask, geometric OOD warnings and
scoped shrinkage-screening fields.

## Training data

The original regime contains 9,710 generated thermal simulations across 42 primitive
geometry families. A later dataset contains 4,727 usable simulations across multiple alloys
and boundary-condition regimes. Targets come from an FEniCSx enthalpy-method solver.

## Evaluation

| Population | n | Nodal MAE | Hot-spot overlap | Field spread |
|---|---:|---:|---:|---:|
| Held-out primitive families | family split | 4.26 s | 0.917 | about 1.02 predicted/reference |
| Real single-body geometry | 6 | 80.3 s median | 0.738 median | 1.000 median |
| Real sleeve assemblies | 2 | 443.7 s median | 0.932 median | 0.227 median |

Two additional real CAD models failed meshing and were not scored. The worst scored
single-body overlap was 0.035. Results are against reference fields produced by the same
physics stack used for training and therefore do not constitute physical validation.

## Known limitations

1. The reference solver and surrogate share property and boundary assumptions.
2. Invalid or highly detailed B-reps can fail or exceed the meshing budget.
3. The default mesh density can under-resolve some large or thin-walled parts.
4. Assembly fields are over-smoothed even when hot-region ranking is good.
5. Geometry-envelope checks do not detect every unfamiliar topology or model failure.
6. Hot spots are shrinkage-risk indicators, not measured defect labels.
7. The chill-bearing Niyama proxy is low-confidence and is flagged as such.

## Recommended interpretation

Use the output as a screening hypothesis: where should an engineer look first, and is the
concept ready for a more complete simulation? Keep the exact mesh, inputs, model version and
warnings with every result. Escalate OOD, low-confidence and high-consequence decisions to a
calibrated simulation and physical review.

## Reproducibility

The trained artifacts and raw data are not distributed in this public snapshot. A complete
internal result records the GNN, total-head and spread-head artifact paths, alloy/mold inputs,
mesh settings and OOD warnings. A versioned manifest with hashes is planned before public
deployment of FoundryFlash.
