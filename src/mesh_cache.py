"""Content-addressed on-disk cache for `src.meshing.heal_and_mesh`.

The key is sha256(STEP bytes) + target node count + `MESH_PIPELINE_VERSION`,
so a byte-identical STEP file is meshed once. Each entry is a pair
`<key>.msh` / `<key>.json` (the mesh and its meta dict), written through a
per-writer temporary file and `os.replace`, so concurrent writers never leave
a half-written entry. Failed meshes are not cached.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid

# MESH_PIPELINE_VERSION and heal_and_mesh are read through globals() so tests
# can monkeypatch them.
from src.meshing import DEFAULT_TARGET_NODES, MESH_PIPELINE_VERSION, heal_and_mesh  # noqa: F401

_HASH_CHUNK_SIZE = 1 << 20  # 1 MiB
# STEP files above this size get the defeaturing attempt first. Changing it
# changes which mesh a file resolves to, so bump MESH_PIPELINE_VERSION too.
_BIG_FILE_THRESHOLD_BYTES = 8 * 1024 * 1024  # 8 MiB


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def mesh_cache_key(step_path: str, target_nodes: int = DEFAULT_TARGET_NODES) -> str:
    """sha256 over the file digest, `target_nodes` and the pipeline version.
    The version is read from module globals so tests can monkeypatch it."""
    digest = _sha256_file(step_path)
    pipeline_version = globals()["MESH_PIPELINE_VERSION"]
    key_input = f"{digest}:{int(target_nodes)}:{int(pipeline_version)}".encode("utf-8")
    return hashlib.sha256(key_input).hexdigest()


def _unique_tmp_path(final_path: str) -> str:
    return f"{final_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def _paths(cache_dir: str, key: str) -> tuple[str, str]:
    return (
        os.path.join(cache_dir, f"{key}.msh"),
        os.path.join(cache_dir, f"{key}.json"),
    )


def get_or_mesh(step_path: str, out_msh: str, target_nodes: int = DEFAULT_TARGET_NODES,
                cache_dir: str = "runs/mesh_cache") -> dict:
    """`heal_and_mesh`'s meta dict for `step_path`, plus a `"cache"` key.

    On a hit the cached mesh is copied to `out_msh` (`"cache": "hit"`). On a
    miss the part is meshed; a successful mesh is stored and returned with
    `"cache": "miss"`, a failed one is returned unchanged and not stored.
    Small files try a plain pass first and retry once with defeaturing; files
    above 8 MiB try the defeatured pass first and retry once without it.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = mesh_cache_key(step_path, target_nodes)
    msh_path, json_path = _paths(cache_dir, key)

    if os.path.exists(msh_path) and os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        shutil.copyfile(msh_path, out_msh)
        result = dict(meta)
        result["cache"] = "hit"
        return result

    heal_and_mesh_fn = globals()["heal_and_mesh"]
    t0 = time.time()
    big_file = os.path.getsize(step_path) > _BIG_FILE_THRESHOLD_BYTES
    if big_file:
        meta = heal_and_mesh_fn(step_path, out_msh, target_nodes=target_nodes, defeature=True)
        defeature_used, retry = True, False
        if not meta.get("ok"):
            meta = heal_and_mesh_fn(step_path, out_msh, target_nodes=target_nodes, defeature=False)
            defeature_used, retry = False, True
    else:
        meta = heal_and_mesh_fn(step_path, out_msh, target_nodes=target_nodes, defeature=False)
        defeature_used, retry = False, False
        if not meta.get("ok"):
            meta = heal_and_mesh_fn(step_path, out_msh, target_nodes=target_nodes, defeature=True)
            defeature_used, retry = True, True
    wall_s = time.time() - t0

    if not meta.get("ok"):
        result = dict(meta)
        result["cache"] = "miss"
        result["defeature"] = defeature_used
        result["retry"] = retry
        return result

    stored_meta = dict(meta)
    stored_meta["mesh_wall_s"] = wall_s
    stored_meta["pipeline_version"] = globals()["MESH_PIPELINE_VERSION"]
    stored_meta["source_path"] = os.path.abspath(step_path)
    stored_meta["created_at"] = time.time()
    stored_meta["defeature"] = defeature_used
    stored_meta["retry"] = retry

    msh_tmp = _unique_tmp_path(msh_path)
    json_tmp = _unique_tmp_path(json_path)
    shutil.copyfile(out_msh, msh_tmp)
    os.replace(msh_tmp, msh_path)
    with open(json_tmp, "w") as f:
        json.dump(stored_meta, f)
    os.replace(json_tmp, json_path)

    result = dict(stored_meta)
    result["cache"] = "miss"
    return result


def clear_mesh_cache(cache_dir: str = "runs/mesh_cache") -> None:
    """Remove the entire cache directory, if present."""
    shutil.rmtree(cache_dir, ignore_errors=True)
