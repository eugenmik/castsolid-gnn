# Model card: solidification surrogate v5.1

The full model card, with architecture, training data, training procedure and quick start,
is on Hugging Face: [eugenmik/castsolid-gnn](https://huggingface.co/eugenmik/castsolid-gnn).
This file is the short version for reviewers of the source code.

## Model details

- Version: v5.1, frozen on 31 July 2026
- Developer: Eugen Miknevic, independent foundry engineer
- Contact: eugenmiknevic@gmail.com
- License: Apache-2.0 for code and weights
- Weights: `weights/gnn_v5.1.safetensors`, `weights/gbm_total_v3.skops`,
  `weights/gbm_std_v2.skops`, `weights/ood_ranges.json`, published on Hugging Face
- Intended role: rapid solidification screening

The predictor is a hybrid of three models. A 12-block graph neural network predicts the spatial
pattern of the field, one gradient-boosted head predicts the total time and another predicts
the spread of the field. A weighted soft-anchor fit combines the three.

## Intended use

- comparing early casting concepts;
- finding candidate last-to-solidify regions;
- deciding which designs need a full process simulation;
- early quoting and design discussions, as long as the limitations stay visible.

## Out-of-scope use

- final process sign-off;
- guaranteed defect or porosity prediction;
- mold filling, flow, gas, stress or distortion analysis;
- unattended decisions on geometry outside the training envelope;
- safety-critical or contractual acceptance decisions without an independent engineering
  review.

## Inputs and outputs

The inputs are a STEP geometry in millimetres, alloy and mold properties, and optional roles for
the bodies of a multi-body STEP (a casting with chills, or a casting with one sleeve). The
outputs are the solidification time per node, the total time, a top-decile hot-spot mask,
geometric OOD warnings and shrinkage screening fields.

## Training data

The original regime has 9,710 generated thermal simulations across 42 primitive geometry
families. A later dataset adds 4,727 usable simulations across multiple alloys and boundary
regimes, and 500 more cases put virtual chill and sleeve patches on real part geometry. An
FEniCSx enthalpy-method solver produced all targets.

## Evaluation

| Population | n | Nodal MAE | Hot-spot overlap | Field spread (predicted/reference) |
|---|---:|---:|---:|---:|
| Held-out primitive families | family split | 4.26 s | 0.917 | about 1.02 |
| Real single-body geometry | 6 | 80.3 s median | 0.738 median | 1.000 median |
| Real sleeve assemblies | 2 | 443.7 s median | 0.932 median | 0.227 median |

Two more real CAD models failed to mesh and were not scored. The worst single-body overlap
was 0.035. The reference fields come from the same physics stack used for training, so these
results are not physical validation.

## Known limitations

1. The reference solver and the surrogate share property and boundary assumptions.
2. Invalid or very detailed B-reps can fail to mesh or run past the meshing time limit.
3. The default mesh density can under-resolve some large or thin-walled parts.
4. Assembly fields are over-smoothed even when the hot regions are ranked well.
5. The geometry-envelope checks do not catch every unfamiliar topology or model failure.
6. Hot spots indicate shrinkage risk. They are not measured defect labels.
7. The Niyama proxy has low confidence on parts with chills and is flagged that way.

## How to read the output

Treat a result as a screening hypothesis: where should an engineer look first, and is the
concept ready for a full simulation? Keep the mesh, inputs, model version and warnings with
every result. Send OOD, low-confidence and high-consequence cases to a calibrated simulation
and a physical review.

## Reproducibility

`tools/export_release.py` converted the internal PyTorch and joblib checkpoints into the
published safetensors and skops files and checked them: every GNN tensor is bit-identical, and
the gradient-boosted heads give identical predictions on probe rows that span their training
split ranges. `weights/SHA256SUMS` lists the checksums. Every result should record the model
version, the alloy and mold inputs, the mesh settings and any OOD warnings.
