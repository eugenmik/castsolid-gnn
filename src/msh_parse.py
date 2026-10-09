"""Reader for gmsh 2.2 ASCII meshes.

Extracts node coordinates (mm, in the file's node order), surface triangles
grouped by physical name ("Cooling", optionally "Adiabatic") and volume
tetrahedra. Connectivity is remapped to 0-based node indices.
"""
from dataclasses import dataclass
import numpy as np

_TRI_TYPE = 2   # 3-node triangle
_TET_TYPE = 4   # 4-node tetrahedron


@dataclass
class Mesh:
    node_xyz: np.ndarray            # (N, 3) float64, mm
    tris: dict                      # name -> (M, 3) int64, 0-based
    tets: np.ndarray                # (K, 4) int64, 0-based


def parse_msh(path: str) -> Mesh:
    lines = open(path).read().splitlines()
    i, n = 0, len(lines)
    phys = {}          # physical tag -> name
    node_ids = []      # id order
    coords = []        # xyz in that order
    tri_by_name = {}   # name -> list of [i,j,k]
    tets = []

    while i < n:
        s = lines[i].strip()
        if s == "$PhysicalNames":
            cnt = int(lines[i + 1])
            for k in range(cnt):
                dim, tag, name = lines[i + 2 + k].split(maxsplit=2)
                phys[int(tag)] = name.strip().strip('"')
            i += 2 + cnt
            continue
        if s == "$Nodes":
            cnt = int(lines[i + 1])
            for k in range(cnt):
                p = lines[i + 2 + k].split()
                node_ids.append(int(p[0]))
                coords.append((float(p[1]), float(p[2]), float(p[3])))
            i += 2 + cnt
            continue
        if s == "$Elements":
            cnt = int(lines[i + 1])
            id2idx = {nid: idx for idx, nid in enumerate(node_ids)}
            for k in range(cnt):
                p = [int(x) for x in lines[i + 2 + k].split()]
                etype, ntags = p[1], p[2]
                tags = p[3:3 + ntags]
                conn = [id2idx[v] for v in p[3 + ntags:]]
                name = phys.get(tags[0]) if ntags else None
                if etype == _TRI_TYPE and name is not None:
                    tri_by_name.setdefault(name, []).append(conn)
                elif etype == _TET_TYPE:
                    tets.append(conn)
            i += 2 + cnt
            continue
        i += 1

    node_xyz = np.asarray(coords, dtype=np.float64)
    tris = {name: np.asarray(v, dtype=np.int64) for name, v in tri_by_name.items()}
    tets_arr = np.asarray(tets, dtype=np.int64) if tets else np.empty((0, 4), np.int64)
    return Mesh(node_xyz=node_xyz, tris=tris, tets=tets_arr)
