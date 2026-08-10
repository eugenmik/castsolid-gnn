import json, os
import pytest
import src.mesh_cache as mc

def _fake_heal(calls, ok=True, n_nodes=5000):
    def fake(step_path, out_msh, target_nodes=8500, **kw):
        calls.append((step_path, target_nodes))
        if ok:
            with open(out_msh, "w") as f:
                f.write("MESH " + str(n_nodes))
        return {"ok": ok, "n_nodes": n_nodes if ok else 0, "notes": "" if ok else "boom"}
    return fake


def _recording_heal(calls, results):
    """A monkeypatch fake that records the `defeature` kwarg passed on each
    call (`calls` accumulates dicts `{"defeature": bool}`) and returns
    `results[i]` for the i-th call (clamped to the last entry once
    exhausted) -- lets a single fake script a "fails plain, succeeds on
    defeature retry" sequence for the trigger-policy tests below."""
    def fake(step_path, out_msh, target_nodes=8500, defeature=False, **kw):
        idx = min(len(calls), len(results) - 1)
        calls.append({"defeature": defeature})
        r = results[idx]
        if r["ok"]:
            with open(out_msh, "w") as f:
                f.write("MESH " + str(r["n_nodes"]))
        return dict(r)
    return fake

def test_miss_then_hit(tmp_path, monkeypatch):
    step = tmp_path / "part.step"; step.write_bytes(b"solid-bytes-1")
    calls = []
    monkeypatch.setattr(mc, "heal_and_mesh", _fake_heal(calls))
    r1 = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"))
    r2 = mc.get_or_mesh(str(step), str(tmp_path / "b.msh"), 8500, cache_dir=str(tmp_path / "c"))
    assert r1["cache"] == "miss" and r2["cache"] == "hit"
    assert len(calls) == 1                                   # second call never meshed
    assert open(tmp_path / "b.msh").read() == open(tmp_path / "a.msh").read()
    assert r2["n_nodes"] == 5000 and r2["pipeline_version"] == mc.MESH_PIPELINE_VERSION

def test_key_changes_with_content_target_and_version(tmp_path, monkeypatch):
    step = tmp_path / "p.step"; step.write_bytes(b"AAA")
    k1 = mc.mesh_cache_key(str(step), 8500)
    step.write_bytes(b"BBB")
    k2 = mc.mesh_cache_key(str(step), 8500)
    k3 = mc.mesh_cache_key(str(step), 5000)
    monkeypatch.setattr(mc, "MESH_PIPELINE_VERSION", mc.MESH_PIPELINE_VERSION + 1)
    k4 = mc.mesh_cache_key(str(step), 8500)
    assert len({k1, k2, k3, k4}) == 4

def test_failure_not_cached(tmp_path, monkeypatch):
    step = tmp_path / "p.step"; step.write_bytes(b"X")
    calls = []
    monkeypatch.setattr(mc, "heal_and_mesh", _fake_heal(calls, ok=False))
    r1 = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"))
    r2 = mc.get_or_mesh(str(step), str(tmp_path / "b.msh"), 8500, cache_dir=str(tmp_path / "c"))
    # `_fake_heal` (ok=False) always fails regardless of `defeature`, so each
    # of the 2 `get_or_mesh` invocations now makes 2 calls under the Task-5
    # auto-trigger policy (plain, then a defeature=True retry) -- 4 total,
    # not 2 (pre-Task-5 baseline, before the retry-on-failure policy existed).
    assert r1["ok"] is False and r2["cache"] == "miss" and len(calls) == 4

# --------------------------------------------------------------------------
# Task 5: size/failure auto-trigger policy for `defeature=`. Big STEP files
# (>8 MiB) go straight to a single defeature=True call (no wasted plain
# attempt on a file already known, empirically, to need it -- see
# task-5-report.md); smaller files try plain first and retry ONCE with
# defeature=True only if the plain attempt fails outright. Either way the
# returned meta records which `defeature` value actually produced the
# result and whether a retry happened, and the cache key is untouched by
# any of this (the policy is a deterministic function of the input file,
# so a given STEP always resolves to the same `defeature` choice).
# --------------------------------------------------------------------------
_BIG_FILE_THRESHOLD_BYTES = 8 * 1024 * 1024


def test_trigger_policy_big_file_single_defeature_call(tmp_path, monkeypatch):
    step = tmp_path / "big.step"
    step.write_bytes(b"x" * (_BIG_FILE_THRESHOLD_BYTES + 1))
    calls = []
    monkeypatch.setattr(mc, "heal_and_mesh",
                         _recording_heal(calls, [{"ok": True, "n_nodes": 999, "notes": ""}]))
    r = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"))
    assert len(calls) == 1
    assert calls[0]["defeature"] is True
    assert r["ok"] is True and r["defeature"] is True and r["retry"] is False


def test_trigger_policy_small_file_ok_plain_no_retry(tmp_path, monkeypatch):
    step = tmp_path / "small.step"
    step.write_bytes(b"tiny")
    calls = []
    monkeypatch.setattr(mc, "heal_and_mesh",
                         _recording_heal(calls, [{"ok": True, "n_nodes": 111, "notes": ""}]))
    r = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"))
    assert len(calls) == 1
    assert calls[0]["defeature"] is False
    assert r["ok"] is True and r["defeature"] is False and r["retry"] is False


def test_trigger_policy_small_file_retries_with_defeature_on_failure(tmp_path, monkeypatch):
    step = tmp_path / "small.step"
    step.write_bytes(b"tiny")
    calls = []
    monkeypatch.setattr(mc, "heal_and_mesh", _recording_heal(calls, [
        {"ok": False, "n_nodes": 0, "notes": "boom"},
        {"ok": True, "n_nodes": 777, "notes": ""},
    ]))
    r = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"))
    assert len(calls) == 2
    assert calls[0]["defeature"] is False
    assert calls[1]["defeature"] is True
    assert r["ok"] is True and r["defeature"] is True and r["retry"] is True
    assert r["n_nodes"] == 777


def test_trigger_policy_big_file_defeature_fails_plain_retry_succeeds(tmp_path, monkeypatch):
    """Symmetric with the small-file retry: a big file's first (defeature=True)
    attempt can also fail outright, and must then retry ONCE with
    defeature=False -- not be left with a single, permanently-failed attempt.
    This must fail on pre-fix code (which had no big-file fallback at all)."""
    step = tmp_path / "big.step"
    step.write_bytes(b"x" * (_BIG_FILE_THRESHOLD_BYTES + 1))
    calls = []
    monkeypatch.setattr(mc, "heal_and_mesh", _recording_heal(calls, [
        {"ok": False, "n_nodes": 0, "notes": "boom"},
        {"ok": True, "n_nodes": 321, "notes": ""},
    ]))
    r = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"))
    assert len(calls) == 2
    assert calls[0]["defeature"] is True
    assert calls[1]["defeature"] is False
    assert r["ok"] is True and r["defeature"] is False and r["retry"] is True
    assert r["n_nodes"] == 321
    assert r["cache"] == "miss"
    # And now it must be cached: a second call must hit, with no further heal_and_mesh calls.
    r2 = mc.get_or_mesh(str(step), str(tmp_path / "b.msh"), 8500, cache_dir=str(tmp_path / "c"))
    assert r2["cache"] == "hit" and len(calls) == 2
    assert r2["defeature"] is False and r2["retry"] is True


def test_trigger_policy_small_file_fails_both_attempts_no_extra_retry(tmp_path, monkeypatch):
    """Even if the defeature retry also fails, `get_or_mesh` must not retry
    a third time -- exactly one retry, ever."""
    step = tmp_path / "small.step"
    step.write_bytes(b"tiny")
    calls = []
    monkeypatch.setattr(mc, "heal_and_mesh", _recording_heal(calls, [
        {"ok": False, "n_nodes": 0, "notes": "boom1"},
        {"ok": False, "n_nodes": 0, "notes": "boom2"},
    ]))
    r = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"))
    assert len(calls) == 2
    assert r["ok"] is False and r["defeature"] is True and r["retry"] is True
    assert r["cache"] == "miss"


def test_tmp_names_are_unique_per_writer(tmp_path):
    # Two concurrent workers racing on the SAME cache miss must never derive
    # the same tmp path from the same final path -- otherwise their writes
    # could interleave on one inode before either os.replace runs, and
    # os.replace would then install a corrupted mesh for every later hit.
    final = str(tmp_path / "c" / "somekey.msh")
    tmp_a = mc._unique_tmp_path(final)
    tmp_b = mc._unique_tmp_path(final)
    assert tmp_a != tmp_b
    assert tmp_a != final + ".tmp" and tmp_b != final + ".tmp"  # no fixed shared name


# --------------------------------------------------------------------------
# Part B Task 4: `keep_solid_id` / `bodies_key` / `post_mesh` (tagged-mesh
# caching) -- MESH_PIPELINE_VERSION pinned to 3.
# --------------------------------------------------------------------------
def test_pipeline_version_pinned_to_3():
    assert mc.MESH_PIPELINE_VERSION == 3


def test_mesh_cache_key_changes_with_bodies_key(tmp_path):
    step = tmp_path / "p.step"; step.write_bytes(b"AAA")
    k_none = mc.mesh_cache_key(str(step), 8500)
    k_default = mc.mesh_cache_key(str(step), 8500, bodies_key="none")
    k_other = mc.mesh_cache_key(str(step), 8500, bodies_key="somehash")
    assert k_none == k_default            # "none" is the implicit default
    assert k_other != k_none


def test_get_or_mesh_forwards_keep_solid_id_to_heal_and_mesh(tmp_path, monkeypatch):
    step = tmp_path / "p.step"; step.write_bytes(b"body-bytes")
    calls = []

    def fake(step_path, out_msh, target_nodes=8500, defeature=False, keep_solid_id=None, **kw):
        calls.append({"keep_solid_id": keep_solid_id, "defeature": defeature})
        with open(out_msh, "w") as f:
            f.write("MESH 42")
        return {"ok": True, "n_nodes": 42, "notes": ""}

    monkeypatch.setattr(mc, "heal_and_mesh", fake)
    r = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"),
                       keep_solid_id=7, bodies_key="rolehash1")
    assert len(calls) == 1 and calls[0]["keep_solid_id"] == 7
    assert r["cache"] == "miss" and r["n_nodes"] == 42

    # A different keep_solid_id / bodies_key must miss the same cache_dir
    # (different key), calling heal_and_mesh again.
    r2 = mc.get_or_mesh(str(step), str(tmp_path / "b.msh"), 8500, cache_dir=str(tmp_path / "c"),
                        keep_solid_id=9, bodies_key="rolehash2")
    assert len(calls) == 2 and calls[1]["keep_solid_id"] == 9


def test_get_or_mesh_post_mesh_hook_runs_on_miss_before_store_and_merges_into_meta(
        tmp_path, monkeypatch):
    step = tmp_path / "p.step"; step.write_bytes(b"tag-bytes")
    heal_calls = []
    post_calls = []

    def fake_heal(step_path, out_msh, target_nodes=8500, defeature=False, keep_solid_id=None, **kw):
        heal_calls.append(1)
        with open(out_msh, "w") as f:
            f.write("UNTAGGED")
        return {"ok": True, "n_nodes": 10, "notes": ""}

    def fake_post_mesh(out_msh):
        post_calls.append(out_msh)
        # simulate tag_contact_patches rewriting the mesh in place
        with open(out_msh, "a") as f:
            f.write(" TAGGED")
        return {"per_body_patch_area_mm2": {2: 12.5}, "n_adiab_tris": 3}

    monkeypatch.setattr(mc, "heal_and_mesh", fake_heal)
    out1 = str(tmp_path / "a.msh")
    r1 = mc.get_or_mesh(str(step), out1, 8500, cache_dir=str(tmp_path / "c"),
                        keep_solid_id=1, bodies_key="bh1", post_mesh=fake_post_mesh)

    assert len(post_calls) == 1 and post_calls[0] == out1
    assert r1["cache"] == "miss"
    assert r1["per_body_patch_area_mm2"] == {2: 12.5}
    assert r1["n_adiab_tris"] == 3
    assert open(out1).read() == "UNTAGGED TAGGED"   # tagged version was stored

    # A cache HIT returns the already-tagged mesh + merged meta, WITHOUT
    # calling post_mesh (or heal_and_mesh) again.
    out2 = str(tmp_path / "b.msh")
    r2 = mc.get_or_mesh(str(step), out2, 8500, cache_dir=str(tmp_path / "c"),
                        keep_solid_id=1, bodies_key="bh1", post_mesh=fake_post_mesh)
    assert r2["cache"] == "hit"
    assert len(post_calls) == 1                      # not called again
    assert len(heal_calls) == 1                       # not re-meshed
    # NOTE: a JSON round-trip through the on-disk cache turns dict int keys
    # into strings (json.dump has no int-key mode) -- a cache HIT's
    # per_body_patch_area_mm2 keys are therefore str, not int, unlike a
    # fresh MISS's. Callers (predict_casting) normalize back to int.
    assert r2["per_body_patch_area_mm2"] == {"2": 12.5}
    assert open(out2).read() == "UNTAGGED TAGGED"     # tagged mesh served from cache


def test_get_or_mesh_no_bodies_key_or_post_mesh_is_byte_identical_to_before(tmp_path, monkeypatch):
    """Back-compat: omitting keep_solid_id/bodies_key/post_mesh entirely
    must behave exactly as before this task (aside from the version-bump
    key change already covered above)."""
    step = tmp_path / "p.step"; step.write_bytes(b"plain-bytes")
    calls = []
    monkeypatch.setattr(mc, "heal_and_mesh", _fake_heal(calls))
    r1 = mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=str(tmp_path / "c"))
    r2 = mc.get_or_mesh(str(step), str(tmp_path / "b.msh"), 8500, cache_dir=str(tmp_path / "c"))
    assert r1["cache"] == "miss" and r2["cache"] == "hit"
    assert "per_body_patch_area_mm2" not in r2


def test_get_or_mesh_post_mesh_exception_propagates_and_stores_nothing(tmp_path, monkeypatch):
    """Review fix (LOW): if `post_mesh` raises (e.g. `tag_contact_patches`
    finds an empty/all-covering patch), `get_or_mesh` must propagate the
    exception AND must not have stored anything under that cache key --
    a subsequent call re-meshes (and re-runs post_mesh) from scratch,
    rather than silently caching an untagged mesh under a bodies-aware key."""
    step = tmp_path / "p.step"; step.write_bytes(b"boom-bytes")
    cache_dir = str(tmp_path / "c")
    heal_calls = []

    def fake_heal(step_path, out_msh, target_nodes=8500, defeature=False, keep_solid_id=None, **kw):
        heal_calls.append(1)
        with open(out_msh, "w") as f:
            f.write("UNTAGGED")
        return {"ok": True, "n_nodes": 10, "notes": ""}

    def failing_post_mesh(out_msh):
        raise ValueError("body 2 does not touch the casting")

    monkeypatch.setattr(mc, "heal_and_mesh", fake_heal)

    with pytest.raises(ValueError, match="does not touch"):
        mc.get_or_mesh(str(step), str(tmp_path / "a.msh"), 8500, cache_dir=cache_dir,
                       keep_solid_id=1, bodies_key="bh-boom", post_mesh=failing_post_mesh)

    assert len(heal_calls) == 1
    # Nothing was stored under this key -- no .msh/.json for it in cache_dir.
    key = mc.mesh_cache_key(str(step), 8500, bodies_key="bh-boom")
    assert not os.path.exists(os.path.join(cache_dir, f"{key}.msh"))
    assert not os.path.exists(os.path.join(cache_dir, f"{key}.json"))

    # A subsequent call re-meshes (and re-runs post_mesh) from scratch --
    # this time succeeding -- rather than serving a stale/partial entry.
    ok_calls = []

    def ok_post_mesh(out_msh):
        ok_calls.append(out_msh)
        return {"per_body_patch_area_mm2": {2: 5.0}, "n_adiab_tris": 1}

    r2 = mc.get_or_mesh(str(step), str(tmp_path / "b.msh"), 8500, cache_dir=cache_dir,
                        keep_solid_id=1, bodies_key="bh-boom", post_mesh=ok_post_mesh)
    assert len(heal_calls) == 2 and len(ok_calls) == 1
    assert r2["cache"] == "miss" and r2["per_body_patch_area_mm2"] == {2: 5.0}
