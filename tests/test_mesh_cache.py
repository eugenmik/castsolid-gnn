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
    # `_fake_heal` (ok=False) always fails, so each of the 2 `get_or_mesh`
    # calls tries twice (plain, then defeature=True): 4 calls in total.
    assert r1["ok"] is False and r2["cache"] == "miss" and len(calls) == 4

# --------------------------------------------------------------------------
# size/failure auto-trigger policy for `defeature=`. Big STEP files
# (>8 MiB) try defeature=True first; smaller files try plain first and retry ONCE with
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


def test_pipeline_version_pinned_to_3():
    assert mc.MESH_PIPELINE_VERSION == 3










