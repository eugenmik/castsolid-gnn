import re
from pathlib import Path

ALL_FACES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
_FACE_RE  = re.compile(r'^[+\-][XYZ]$')

# Integer encoding for geometry type (used as global feature)
GEOMETRY_ENCODING = {
    'BC':  0, 'RIB': 1, 'L':   2, 'T':   3, 'X':   4,
    'WL':  5, 'UL':  6, 'UM':  7, 'US':  8, 'BH':  9,
    'BO': 10, 'PB': 11, 'PF': 12, 'PH': 13,
    'RPB':14, 'RPF':15, 'RPH':16,
    # everything else → 17 (other)
}
GEOMETRY_OTHER = 17


def parse_filename(path: str) -> dict:
    """Parse any <PREFIX>_<tokens...>.pt filename used in the Foundry dataset.

    Token classification (applied to tokens after the prefix):
      - numeric (all digits)  → appended to dims list
      - matches [+-][XYZ]     → adiabatic face
      - anything else         → ignored (angle suffixes like 30, 45, 135
                                 are already absorbed into the prefix token)

    Returns:
        prefix          — geometry type string (e.g. 'BC', 'RIB', 'L')
        dims            — list of numeric dim values in mm  (may be 0–3 ints)
        adiabatic_faces — list of face labels that are adiabatic
        cooling_faces   — complement of adiabatic_faces over ALL_FACES
        volume_mm3      — product of dims (mm³); 0.0 if no dims
        geometry_type   — integer encoding (GEOMETRY_ENCODING, 17 = other)
    """
    stem   = Path(path).stem
    tokens = stem.split('_')
    prefix = tokens[0]

    dims       = []
    adiabatic  = []
    for tok in tokens[1:]:
        if tok.isdigit():
            dims.append(int(tok))
        elif _FACE_RE.match(tok):
            adiabatic.append(tok)
        # else: skip (e.g. stray letters, already part of prefix)

    cooling  = [f for f in ALL_FACES if f not in set(adiabatic)]
    volume   = float(eval('*'.join(str(d) for d in dims))) if dims else 0.0
    geo_type = GEOMETRY_ENCODING.get(prefix, GEOMETRY_OTHER)

    return {
        'prefix':          prefix,
        'dims':            dims,
        'adiabatic_faces': adiabatic,
        'cooling_faces':   cooling,
        'volume_mm3':      volume,
        'geometry_type':   geo_type,
    }


def global_feature_vec(info: dict) -> list:
    """6-element global feature vector from a parse_filename result.

    Layout: [dim1, dim2, dim3, volume_mm3, n_cooling_faces, geometry_type]
    dims are zero-padded to length 3.
    """
    d = (info['dims'] + [0, 0, 0])[:3]
    return [
        float(d[0]),
        float(d[1]),
        float(d[2]),
        info['volume_mm3'],
        float(len(info['cooling_faces'])),
        float(info['geometry_type']),
    ]


if __name__ == '__main__':
    test_cases = [
        # BC — 3 numeric dims
        'BC_06_08_14_+X_+Z.pt',
        'BC_06_08_14.pt',
        # RIB — 3 dims
        'RIB_06_06_06.pt',
        # L — 2 dims
        'L_06_06.pt',
        # T — 2 dims with faces
        'T_06_08_+X_+Y.pt',
        # X — 2 dims
        'X_06_06_+Y.pt',
        # WL — 2 dims
        'WL_06_06.pt',
        # BH — 2 dims
        'BH_06_06_+X.pt',
        # CYL — 1 dim
        'CYL_06.pt',
        # KON — 1 dim
        'KON_25.pt',
        # angle-suffix prefix
        'L135_06_06.pt',
        'TR30_06_06.pt',
    ]

    header = f"{'File':<35s} {'prefix':<8} {'dims':<14} {'adiab':<18} {'vol':>8} {'geo':>4}"
    print(header)
    print('-' * len(header))
    for p in test_cases:
        r = parse_filename(p)
        gf = global_feature_vec(r)
        print(
            f"{Path(p).name:<35s} {r['prefix']:<8} {str(r['dims']):<14} "
            f"{str(r['adiabatic_faces']):<18} {r['volume_mm3']:>8.0f} {r['geometry_type']:>4d}"
        )
        print(f"  global_features: {gf}")
