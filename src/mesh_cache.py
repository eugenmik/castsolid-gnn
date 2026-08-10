"""Content-keyed mesh cache for the STEP -> tet-mesh pipeline.

Inference meshing (`tools.doe.prep_parts.heal_and_mesh`) takes minutes and is
routinely repeated for byte-identical STEP inputs (re-runs, batch backends,
concurrent workers). This module memoizes that call on disk, keyed on the
SHA-256 of the STEP file's bytes + `target_nodes` + `MESH_PIPELINE_VERSION`,
so any change to the meshing pipeline invalidates the cache automatically
(bump `MESH_PIPELINE_VERSION` in `tools/doe/prep_parts.py` whenever the mesh
output could change).

`heal_and_mesh` and `MESH_PIPELINE_VERSION` are imported at module level and
referenced dynamically (via `globals()`/module attribute lookup) so tests can
monkeypatch `mesh_cache.heal_and_mesh` / `mesh_cache.MESH_PIPELINE_VERSION`
without needing a real gmsh install.

Cache layout (under `cache_dir`, default `runs/mesh_cache/`):
    <key>.msh   -- the cached gmsh 2.2 mesh (only written when ok=True)
    <key>.json  -- meta dict (ok, n_nodes, ..., mesh_wall_s, pipeline_version,
                   source_path, created_at)

Writes are atomic and per-writer-unique (`<key>.msh.<pid>.<uuid>.tmp` /
`<key>.json.<pid>.<uuid>.tmp`, then `os.replace` onto the shared final path)
so concurrent workers racing on the same cache miss never share a tmp inode
(which would let their writes interleave and corrupt it) and never observe a
partially written file -- whichever replace lands last simply wins, and both
writers produced an equally valid, uncorrupted mesh for the same key.

A failed `heal_and_mesh` call (`ok=False`) is never cached: a fixed STEP file
(or a differently-tuned pipeline, which would change the key anyway) must be
retried, not permanently remembered as unmeshable.

Defeaturing auto-trigger (mesh-accel Task 5, 2026-07-29): `get_or_mesh`
decides whether to pass `defeature=True` to `heal_and_mesh` via a
deterministic, input-only policy, symmetric in both directions -- a STEP
file whose on-disk byte size exceeds `_BIG_FILE_THRESHOLD_BYTES` (8 MiB)
goes straight to a `defeature=True` attempt first (large real parts are,
empirically, the ones that need it -- see
`tools/doe/prep_parts.py::_apply_defeature`), and
only if that attempt fails outright (`ok=False`) retries ONCE more with
`defeature=False`; a smaller file does the mirror image -- plain
(`defeature=False`) first, retrying ONCE with `defeature=True` only on
outright failure. Either way there is at most one retry, ever. Because the
policy is a pure function of the input (file size, then a same-input
retry-on-failure), a given STEP file always resolves to the same
`defeature` choice across runs/workers -- so `defeature` does NOT need to
be (and is not) part of `mesh_cache_key`; the cache key stays exactly as
before this task. The returned meta dict always carries `defeature` (bool,
whichever value actually produced the returned result) and `retry` (bool,
True only when a second attempt actually ran, i.e. the first attempt --
big-file `defeature=True` or small-file `defeature=False` -- failed
outright; False whenever the first attempt already succeeded or the input
size never triggers the alternate branch).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid

from tools.doe.prep_parts import DEFAULT_TARGET_NODES, MESH_PIPELINE_VERSION, heal_and_mesh

_HASH_CHUNK_SIZE = 1 << 20  # 1 MiB
# Participates in mesh-output selection (which `defeature` attempt is tried
# first -- see the "Defeaturing auto-trigger" section above). It is NOT part
# of `mesh_cache_key`, so changing this value changes what mesh a given STEP
# file resolves to WITHOUT changing its cache key -- any change here requires
# bumping `MESH_PIPELINE_VERSION` (tools/doe/prep_parts.py) to invalidate
# stale cache entries. Cross-file contract; keep in sync with prep_parts.py.
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


def mesh_cache_key(step_path: str, target_nodes: int = DEFAULT_TARGET_NODES,
                   bodies_key: str = "none") -> str:
    """Content-addressed cache key: sha256(file bytes) + target_nodes +
    MESH_PIPELINE_VERSION + bodies_key, so the key changes if any of those
    change. Reads `MESH_PIPELINE_VERSION` dynamically from module globals so
    tests can monkeypatch it.

    `bodies_key` (Part B Task 4): a caller-supplied string folding in the
    Part-B multi-body role assignment (see
    `src.predict_casting._bodies_cache_key`) -- default `"none"` for the
    ordinary single-body inference path, so an ordinary call's key is
    unaffected by this parameter's mere existence (a bodies-aware and a
    bodies-less call for the SAME step_path/target_nodes only collide if
    the bodies-aware caller explicitly passes `bodies_key="none"`, which it
    never does -- `_bodies_cache_key` always produces a real hash).
    """
    digest = _sha256_file(step_path)
    pipeline_version = globals()["MESH_PIPELINE_VERSION"]
    key_input = f"{digest}:{int(target_nodes)}:{int(pipeline_version)}:{bodies_key}".encode("utf-8")
    return hashlib.sha256(key_input).hexdigest()


def _unique_tmp_path(final_path: str) -> str:
    """A per-writer-unique tmp sibling of `final_path`. Two workers racing on
    the same cache miss must never write the same tmp inode (that would let
    their writes interleave and corrupt it before either `os.replace` runs) --
    each caller gets its own pid+uuid-suffixed tmp file, then atomically
    replaces the SAME shared final path (last replace wins, and both writers
    produced a fully-formed, uncorrupted file)."""
    return f"{final_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def _paths(cache_dir: str, key: str) -> tuple[str, str]:
    return (
        os.path.join(cache_dir, f"{key}.msh"),
        os.path.join(cache_dir, f"{key}.json"),
    )


def get_or_mesh(step_path: str, out_msh: str, target_nodes: int = DEFAULT_TARGET_NODES,
                cache_dir: str = "runs/mesh_cache", keep_solid_id: int | None = None,
                bodies_key: str = "none", post_mesh=None) -> dict:
    """Return a `heal_and_mesh`-shaped meta dict for `step_path`, using the
    on-disk cache when possible.

    - Hit: copy the cached `.msh` to `out_msh`, return the stored meta dict
      plus `{"cache": "hit"}` (no `heal_and_mesh` call, no `post_mesh`
      call -- the cached mesh is already whatever `post_mesh` produced on
      the miss that created it, see below).
    - Miss + `ok=True`: call `heal_and_mesh` (forwarding `keep_solid_id`),
      then -- if `post_mesh` is given -- call `post_mesh(out_msh) -> dict`
      and merge its return into the meta, THEN atomically store both the
      (now possibly `post_mesh`-modified) mesh and its meta json under the
      cache key, and return the meta dict plus `{"cache": "miss"}`.
    - Miss + `ok=False`: store nothing, return `heal_and_mesh`'s dict
      unchanged plus `{"cache": "miss"}` -- failures are never cached, and
      `post_mesh` is never called on a failed mesh.

    `keep_solid_id` (Part B Task 4): the user-designated casting solid, when
    a multi-body STEP's role assignment has already picked which solid IS
    the casting -- forwarded verbatim to `heal_and_mesh`, in place of the
    largest-solid default (`None`, unchanged).

    `bodies_key` (Part B Task 4): folded into `mesh_cache_key` (see there)
    so a role-assignment change (a different `bodies.json`, a registry edit
    that changes a resolved BC scalar, etc.) is a genuinely different cache
    entry, never a stale hit.

    `post_mesh` (Part B Task 4): an optional `(out_msh) -> dict` hook run
    on a cache MISS, after `heal_and_mesh` succeeds and BEFORE the mesh is
    stored -- e.g. `src.assembly.tag_contact_patches`, which rewrites
    `out_msh` in place (Cooling -> Adiabatic contact patch) and returns
    `{"per_body_patch_area_mm2": ..., "n_adiab_tris": ...}`. Tagging cost is
    not trivial on big meshes, so the TAGGED mesh is what gets cached (the
    bodies hash is already in the key) -- a later hit for the same
    `bodies_key` returns the tagged mesh directly, with no re-tagging.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = mesh_cache_key(step_path, target_nodes, bodies_key)
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
        # Large STEP files go straight to the defeatured attempt -- no
        # point burning a near-certain-to-fail plain pass first. But if
        # even that fails outright, retry ONCE with defeature=False --
        # symmetric with the small-file policy below, so a big file is
        # never left with only one attempt while a small file gets two.
        meta = heal_and_mesh_fn(step_path, out_msh, target_nodes=target_nodes, defeature=True,
                                keep_solid_id=keep_solid_id)
        defeature_used, retry = True, False
        if not meta.get("ok"):
            meta = heal_and_mesh_fn(step_path, out_msh, target_nodes=target_nodes, defeature=False,
                                    keep_solid_id=keep_solid_id)
            defeature_used, retry = False, True
    else:
        meta = heal_and_mesh_fn(step_path, out_msh, target_nodes=target_nodes, defeature=False,
                                keep_solid_id=keep_solid_id)
        defeature_used, retry = False, False
        if not meta.get("ok"):
            # Exactly one retry, ever -- a defeature=True failure is final.
            meta = heal_and_mesh_fn(step_path, out_msh, target_nodes=target_nodes, defeature=True,
                                    keep_solid_id=keep_solid_id)
            defeature_used, retry = True, True
    wall_s = time.time() - t0

    if not meta.get("ok"):
        result = dict(meta)
        result["cache"] = "miss"
        result["defeature"] = defeature_used
        result["retry"] = retry
        return result

    post_meta = post_mesh(out_msh) if post_mesh is not None else {}

    stored_meta = dict(meta)
    stored_meta.update(post_meta)
    stored_meta["mesh_wall_s"] = wall_s
    stored_meta["pipeline_version"] = globals()["MESH_PIPELINE_VERSION"]
    stored_meta["source_path"] = os.path.abspath(step_path)
    stored_meta["created_at"] = time.time()
    stored_meta["defeature"] = defeature_used
    stored_meta["retry"] = retry

    msh_tmp = _unique_tmp_path(msh_path)
    json_tmp = _unique_tmp_path(json_path)
    shutil.copyfile(out_msh, msh_tmp)  # copies the (possibly post_mesh-tagged) mesh
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
