# Public runtime property tables

These compact CSV files contain the numeric rows required by the public inference and unit-test
code. They are a curated runtime subset of the project's larger property record.

- `materials.csv`: 12 named iron and steel grades.
- `molds.csv`: two UI labels currently sharing the same interim thermal properties.
- `chills.csv`: example chill materials; incomplete rows are skipped by the loader.
- `exo_calibration.csv`: the MEF-to-heat-flux calibration used by the example sleeve path.

The values remain modeling inputs, not certified material specifications. A production user
should replace or verify them for the actual alloy, mold and process and should retain the exact
property row with every result.
