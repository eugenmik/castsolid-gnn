# Property tables

These two CSV files hold the material and mold rows that inference reads.

- `materials.csv` has two grades: `EN-GJS-400-15` (ductile iron, the default) and `G20Mn5`
  (cast steel). Columns: thermal conductivity `k_W_mK`, density `rho_kg_m3`, specific heat
  `cp_J_kgK`, latent heat `L_kJ_kg`, liquidus `TL_C`, solidus `TS_C` and the pouring
  temperature range `Tpour_lo_C`/`Tpour_hi_C`. The prediction uses the midpoint of that range.
- `molds.csv` has one mold system, `green_sand`. The model reads its thermal effusivity
  (`effusivity_W_s05_m2K`), which is the square root of the product of the conductivity,
  density and specific heat in the same row.

The values come from handbook data. They are modelling inputs, not certified material
specifications. Check them against your actual alloy and process, and keep the exact row you
used with every result.

To try an alloy that is not listed, pass the properties directly instead of editing the table:

```bash
python -m src.predict_casting part.step --props my_alloy.json
```

`my_alloy.json` needs exactly these keys: `k`, `rho`, `cp`, `L_kJ_kg`, `TL`, `TS`, `Tpour`,
`mold_effusivity`. The model was trained on iron and steel grades in sand molds; properties far
from that range are extrapolation.
