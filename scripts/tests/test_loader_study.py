import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import pytest

STUDY = Path(__file__).resolve().parents[2] / "benchmarks" / "loader"
if not STUDY.is_dir():
    pytest.skip("developer study sources are not shipped in the sdist", allow_module_level=True)
sys.path.insert(0, str(STUDY))
import corpus
import report
import run


@pytest.fixture
def config():
    return run.load_config(STUDY / "study.json")


def test_frozen_config_and_worker_profiles(config):
    run.validate_config(config)
    assert len(run.configurations(config)) == 18
    assert run.pipeline_kwargs(config, 0)["prefetch_factor"] is None
    assert run.pipeline_kwargs(config, 2)["multiprocessing_context"] == "spawn"
    for profile_index in (0, 1):
        for key, bad in (("prefetch_factor", 2 if profile_index == 0 else None),
                         ("persistent_workers", profile_index == 0),
                         ("multiprocessing_context", "spawn" if profile_index == 0 else None)):
            changed = copy.deepcopy(config)
            changed["pipeline"]["worker_profiles"][profile_index][key] = bad
            with pytest.raises(ValueError):
                run.validate_config(changed)
            with pytest.raises(ValueError):
                run.pipeline_kwargs(changed, profile_index * 2)
    for key in list(config):
        changed = copy.deepcopy(config)
        del changed[key]
        with pytest.raises(ValueError):
            run.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["unknown"] = 1
    with pytest.raises(ValueError):
        run.validate_config(changed)


def evidence(config):
    validation = {"binding": {"sha": "a", "corpus": "b"}, "order_digest": "order", "cells": {}}
    rows = []
    for cell in run.configurations(config):
        validation["cells"][cell["config_id"]] = {"result_digest": "pixels"}
        for round_ in range(5):
            for epoch in range(2):
                rows.append(dict(cell, round=round_, epoch=epoch, elapsed_ns=1_200_000_000,
                                 samples=1000, startup_ns=100, preparation_ns=200,
                                 binding=validation["binding"], order_digest="order", result_digest="pixels"))
    return validation, rows


def test_aggregation_and_negative_results(config):
    validation, rows = evidence(config)
    result = report.timings(config, rows, validation)
    assert len(result["configurations"]) == 18
    for effect in result["effects"].values():
        assert not effect["reliable_acceleration"] and not effect["material_regression"]
        assert effect["break_even_epochs"] == [None] * 5
    assert report.distribution([1, 2, 3, 4, 5]) == {"values": [1, 2, 3, 4, 5], "median": 3, "iqr": 2, "min": 1, "max": 5}
    assert report.break_even(200, 100, 8, 10) == 50
    assert report.break_even(50, 100, 8, 10) == 0
    assert report.break_even(200, 100, 10, 10) is None
    assert report.compare([1.06] * 5, [1] * 5, config["decision"])["material_regression"]
    assert not report.compare([1.06] * 3 + [0.9] * 2, [1] * 5, config["decision"])["material_regression"]
    assert report.compare([0.97] * 5, [1] * 5, config["decision"])["reliable_acceleration"]


@pytest.mark.parametrize("change", ["missing", "short", "sha", "order", "pixels", "samples", "duplicate"])
def test_reject_invalid_timing(config, change):
    validation, rows = evidence(config)
    if change == "missing":
        rows.pop()
    elif change == "short":
        rows[0]["elapsed_ns"] = 1
    elif change == "sha":
        rows[0]["binding"] = {"sha": "wrong"}
    elif change == "order":
        rows[0]["order_digest"] = "wrong"
    elif change == "pixels":
        rows[0]["result_digest"] = "wrong"
    elif change == "samples":
        rows[0]["samples"] = 1
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        report.timings(config, rows, validation)


def test_manifest_recipe():
    values = corpus.manifest_paths(1000)
    assert len(set(values)) == 1000
    assert all(len(path.encode("ascii")) == 64 for path in values)
    assert values[0] == "manifest/000000/" + "x" * 44 + ".jpg"


def test_native_attribution_counts_each_stack_once(tmp_path):
    path = tmp_path / "folded.stacks"
    path.write_text("python;tj3Init;alloc_small (jmemmgr.c);malloc 100\npython;malloc 1000\npython;tj3Decompress8;jpeg_get_large;malloc 300\n")
    result = report.native_bytes(path)
    assert result["native_bytes"] == 400 and result["all_bytes"] == 1400
    assert sum(result["sites"].values()) == 400


def test_stage_integrity_and_retry(config, tmp_path):
    path = tmp_path / "evidence"
    path.write_text("one")
    record = {"exit_code": 0, "binding": {"sha": "x"}, "artifacts": {"evidence": corpus.digest(path)}}
    run.verify_stage(record, tmp_path, {"sha": "x"})
    path.write_text("two")
    with pytest.raises(ValueError):
        run.verify_stage(record, tmp_path, {"sha": "x"})
    control = tmp_path / "control.attempt-02"
    previous = tmp_path / "control.attempt-01"
    previous.mkdir()
    for failure in ("external-interruption", "semantic-or-unclassified"):
        (previous / "completion.json").write_text(json.dumps({"success": False, "failure_kind": failure, "elapsed_ns": 1000}))
        if failure == "external-interruption":
            run.check_retry(config, control, tmp_path / "attempt-02")
        else:
            with pytest.raises(ValueError):
                run.check_retry(config, control, tmp_path / "attempt-02")


def test_cancel_child_group(tmp_path):
    pid_path = tmp_path / "child.pid"
    code = "import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(30)" % str(pid_path)
    with (tmp_path / "out").open("w") as out, (tmp_path / "err").open("w") as err:
        with pytest.raises(TimeoutError):
            run.process_group([sys.executable, "-c", code], out, err, 0.4)
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_help_and_config_only_commands():
    for script, flag in (("run.py", "--help"), ("report.py", "--help"), ("run.py", "--self-check")):
        result = subprocess.run([sys.executable, str(STUDY / script), flag], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr


def test_fixed_corrupt_fixture_and_tiny_factory_smoke(config, tmp_path):
    """Correctness smoke only: no study timing or statistical result is emitted."""
    imgread = pytest.importorskip("imgread", reason="native correctness smoke requires an installed wheel")
    import numpy as np
    from PIL import Image
    pixels = np.random.default_rng(37).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / "valid.jpg"
    Image.fromarray(pixels).save(path, quality=95, subsampling=0)
    corrupt = bytearray(path.read_bytes())
    sos = corrupt.index(b"\xff\xda")
    entropy = sos + 2 + int.from_bytes(corrupt[sos + 2:sos + 4], "big")
    corrupt[entropy + 16:entropy + 20] = b"\xff\xc4\x00\x01"
    invalid = tmp_path / "invalid.jpg"
    invalid.write_bytes(corrupt)
    for backend in ("auto", "image"):
        with pytest.raises(RuntimeError):
            imgread.load_numpy(invalid, backend=backend)
        results = []
        for variant in config["variants"]:
            dataset = run.StudyDataset([(str(path), 3)], variant, backend)
            result = run.epoch(dataset, None, [0, 0], validate=True)
            results.append(result)
        assert len(set(results)) == 1


def test_pipeline_and_memory_factory_smoke(config, tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("isolated torch environment only")
    import imgread
    if not hasattr(imgread.Loader, "_debug_state"):
        pytest.skip("diagnostic wheel required for memory smoke")
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    torch.set_num_threads(1)
    path = tmp_path / "small.png"
    Image.new("RGB", (256, 256), (1, 20, 50)).save(path)
    samples = [(str(path), 7), (str(path), 7)]
    results = []
    for cell in run.configurations(config):
        if cell["workload"] != "pipeline":
            continue
        dataset, batches, _, _ = run.make_work(config, cell, samples, [1, 0, 1])
        try:
            results.append(run.epoch(dataset, batches, [1, 0, 1], validate=True))
        finally:
            run.close_batches(batches)
    assert len(set(results)) == 1
    # Reduced correctness-only sequence verifies the actual phase barriers and
    # persistent worker ownership without running the scientific memory matrix.
    small = copy.deepcopy(config)
    small["sequence"]["phases"] = [["small-before", "small", 1]]
    for workers in (0, 2):
        result = run.sequence_job(small, {"backend": "image", "cap": 1048576, "workers": workers}, {"small": {"path": str(path)}})
        active = [row for row in result["rows"] if row["role"] == ("worker" if workers else "parent")]
        assert len({row["pid"] for row in active}) == max(1, workers)
        assert len(active) == 5 * max(1, workers)
        assert {row["phase"] for row in active} == {"small-before", "outputs-retained", "outputs-released", "destroyed"}
    for variant in config["variants"]:
        for method in ("spawn", "forkserver", "fork"):
            if method not in run.mp.get_all_start_methods():
                continue
            result = run.manifest_job(config, {"length": 10, "variant": variant, "workers": 2, "method": method})
            assert len(result["rows"]) == 6


def test_manifest_first_worker_survives_slow_sibling_start(config, monkeypatch):
    """A ready worker must wait while its sibling is being spawned/unpickled."""
    pytest.importorskip("imgread")
    context = run.mp.get_context("spawn")
    parents = []
    def pipe():
        parent, child = context.Pipe()
        parents.append(parent)
        return parent, child
    def process(*args, **kwargs):
        child = context.Process(*args, **kwargs)
        original_start = child.start
        if len(parents) == 2:
            def slow_start():
                assert parents[0].poll(20), "first worker did not become ready"
                time.sleep(5.2)
                del child.start  # The actual process object must remain pickleable.
                original_start()
            child.start = slow_start
        return child
    monkeypatch.setattr(run.mp, "get_context", lambda _method: types.SimpleNamespace(Pipe=pipe, Process=process))
    result = run.manifest_job(config, {"length": 2, "variant": "function", "workers": 2, "method": "spawn"})
    workers = [row for row in result["rows"] if row["role"] == "worker"]
    assert len(workers) == 4
    assert len({row["pid"] for row in workers}) == 2
    assert {row["phase"] for row in workers} == {0, 1}


def test_manifest_phase_waits_do_not_reset_deadline(monkeypatch):
    clock = [100.0]
    waits = []
    class Connection:
        def poll(self, timeout):
            waits.append(timeout)
            clock[0] += 7
            return True
        def recv(self):
            return "ready"
    monkeypatch.setattr(run.time, "monotonic", lambda: clock[0])
    connection = Connection()
    assert run.manifest_receive(connection, 112, "phase") == "ready"
    assert run.manifest_receive(connection, 112, "phase") == "ready"
    with pytest.raises(TimeoutError, match="phase"):
        run.manifest_receive(connection, 112, "phase")
    assert waits == [12.0, 5.0]


@pytest.mark.skipif(os.environ.get("IMGREAD_REQUIRE_LARGE_MANIFEST") != "1", reason="explicit million-entry spawn regression")
def test_million_entry_index_manifest_spawn(config):
    import imgread
    if not hasattr(imgread.Loader, "_debug_state"):
        pytest.fail("large manifest regression requires a diagnostic wheel")
    previous_affinity = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, config["environment"]["cpus"])
        result = run.manifest_job(config, {"length": 1_000_000, "variant": "index", "workers": 2, "method": "spawn"})
    finally:
        os.sched_setaffinity(0, previous_affinity)
    assert len(result["rows"]) == 6
    workers = [row for row in result["rows"] if row["role"] == "worker"]
    assert len({row["pid"] for row in workers}) == 2
    for row in result["rows"]:
        assert row["retained_python_paths"] == row["diagnostics"]["manifest_entries"] == 1_000_000
        assert row["diagnostics"]["input_capacity"] == row["diagnostics"]["native_live"] == 0


def test_complete_memory_matrix_and_missing_phase(config, tmp_path):
    rows = []
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    counter = 0
    def sampling():
        nonlocal counter
        counter += 1
        (jobs / f"{counter}.samples.jsonl").write_text(json.dumps({"pid": counter, "sample_ns": 1, "rss_bytes": 100, "pss_bytes": 80, "peak_rss_bytes": 100}) + "\n")
    def state(entries=0, live=0):
        return {"input_len": 0, "input_capacity": 0, "input_peak_capacity": 10, "native_live": live, "native_creations": live,
                "manifest_entries": entries, "manifest_bytes": entries * 100}
    for length in config["manifest"]["lengths"]:
        for workers in (0, 2):
            for method in ([m for m in config["manifest"]["start_methods"] if m in run.mp.get_all_start_methods()] if workers else [None]):
                for variant in config["variants"]:
                    for repeat in range(3):
                        sampling()
                        for phase in range(2):
                            for worker in range(1 + workers):
                                rows.append(dict(kind="manifest", length=length, workers=workers, method=method, variant=variant, repeat=repeat,
                                                 pid=worker + 1, phase=phase, role="parent" if worker == 0 else "worker", rss_bytes=100, pss_bytes=80, peak_rss_bytes=100,
                                                 diagnostics=state(length if variant == "index" else 0),
                                                 preparation={key: 100 for key in ("list_ns", "construction_ns", "pickle_bytes", "pickle_ns", "startup_ns")}))
    for backend in config["backends"]:
        for cap in (0, 1048576):
            for workers in (0, 2):
                sampling()
                for epoch in range(2):
                    phases = [(phase, count) for phase, _kind, count in config["sequence"]["phases"]]
                    if epoch == 1:
                        phases += [(phase, 0) for phase in ("outputs-retained", "outputs-released", "destroyed")]
                    for phase, count in phases:
                        for worker in range(1 + workers):
                            row = dict(kind="sequence", backend=backend, cap=cap, workers=workers, epoch=epoch, phase=phase,
                                       pid=worker + 1, role="parent" if worker == 0 else "worker", rss_bytes=100, pss_bytes=80, peak_rss_bytes=100)
                            if workers == 0 or worker > 0:
                                row["diagnostics"] = state(live=int(backend == "auto" and phase != "destroyed"))
                                row["calls"] = [{"diagnostics": state(live=int(backend == "auto"))} for _ in range(count)]
                                row["output_count"] = 8 if phase == "outputs-retained" else 0
                                row["output_nbytes"] = row["output_count"] * 64 * 64 * 3
                            rows.append(row)
    result = report.memory_evidence(config, rows, tmp_path)
    assert result["sequence_cases"] == 8 and result["manifest_cases"] > 0
    assert len(result["manifest_summary"]) > 0
    with pytest.raises(ValueError, match="incomplete"):
        report.memory_evidence(config, rows[:-1], tmp_path)
    wrong_outputs = copy.deepcopy(rows)
    next(row for row in wrong_outputs if row.get("phase") == "outputs-retained" and "output_nbytes" in row)["output_nbytes"] = 8 * 256 * 256 * 3
    with pytest.raises(ValueError, match="retained output control mismatch"):
        report.memory_evidence(config, wrong_outputs, tmp_path)


def test_native_matrix_allows_zero_native_at_global_peak(config, tmp_path):
    directory = tmp_path / "native"
    directory.mkdir()
    manifest = []
    phases = [phase[0] for phase in config["sequence"]["phases"]]
    for checkpoint in config["native"]["checkpoints"]:
        for repeat in (1, 2, 3):
            stem = directory / f"{checkpoint}-{repeat:02d}"
            profile = Path(str(stem) + ".gz")
            profile.write_bytes(b"schema-only fixture; no profiler invocation")
            exports = {}
            for label in ("live", "peak"):
                path = Path(str(stem) + f".{label}.stacks")
                has_native = not ((label == "peak" and checkpoint == "small-before") or (label == "live" and checkpoint in ("corrupt", "destroyed")))
                path.write_text("python;malloc 1000\n" + ("python;tj3Init;malloc 100\n" if has_native else ""))
                exports[label] = {"path": str(path), "sha256": corpus.digest(path)}
            prefix = phases if checkpoint == "destroyed" else phases[:phases.index(checkpoint) + 1]
            Path(str(stem) + ".diagnostics.json").write_text(json.dumps({"checkpoint": checkpoint, "repeat": repeat,
                "records": [{"phase": phase, "diagnostics": {"input_len": 0, "input_capacity": 0, "native_live": int(phase != "corrupt")}} for phase in prefix]}))
            manifest.append({"checkpoint": checkpoint, "repeat": repeat, "profile": str(profile), "profile_sha256": corpus.digest(profile), "exports": exports})
    (tmp_path / "native-profiles.json").write_text(json.dumps(manifest))
    assert len(report.native_evidence(config, tmp_path)) == 24
    (tmp_path / "native-profiles.json").write_text(json.dumps(manifest[:-1]))
    with pytest.raises(ValueError, match="incomplete"):
        report.native_evidence(config, tmp_path)


def test_internal_worker_request_cannot_escape_or_override_matrix(config, tmp_path):
    (tmp_path / "jobs").mkdir()
    request = {"kind": "timing", "cell": run.configurations(config)[0], "repetitions": 2,
               "result_path": str(tmp_path / "jobs" / "test.result.json")}
    run.validate_request(config, request, tmp_path)
    with pytest.raises(ValueError):
        run.validate_request(config, dict(request, result_path=str(tmp_path / "escape.json")), tmp_path)
    changed = copy.deepcopy(request)
    changed["cell"]["backend"] = "turbojpeg"
    with pytest.raises(ValueError):
        run.validate_request(config, changed, tmp_path)


def test_progressive_fixture_encoder_scratch(config, tmp_path):
    pytest.importorskip("imgread", reason="fixture correctness requires an installed wheel")
    import io
    import numpy as np
    import PIL
    from PIL import Image, ImageFile
    settings = config["sequence"]
    assert settings["small_shape"] == [64, 64, 3]
    assert settings["large_shape"] == [2048, 2048, 3]
    previous = ImageFile.MAXBLOCK
    generated = corpus.generate(settings, tmp_path)
    assert ImageFile.MAXBLOCK == previous
    assert generated["versions"] == {"numpy": np.__version__, "pillow": PIL.__version__}
    assert set(generated["inputs"]) == {"small", "small-progressive", "small-png", "large", "progressive", "large-png", "corrupt", "oversized"}
    for size, shape in (("small", (64, 64, 3)), ("large", (2048, 2048, 3))):
        expected = np.random.Generator(np.random.PCG64(37)).integers(0, 256, shape, dtype=np.uint8)
        with Image.open(generated["inputs"][size + "-png"]["path"]) as png:
            np.testing.assert_array_equal(np.asarray(png), expected)
        baseline = io.BytesIO()
        Image.fromarray(expected).save(baseline, format="JPEG", quality=95, subsampling=0, optimize=False, progressive=False)
        assert Path(generated["inputs"][size]["path"]).read_bytes() == baseline.getvalue()
        progressive = "small-progressive" if size == "small" else "progressive"
        with Image.open(generated["inputs"][progressive]["path"]) as jpg:
            assert jpg.info["progressive"]
    black = io.BytesIO()
    Image.new("RGB", (2, 2), (0, 0, 0)).save(black, format="JPEG", quality=95, subsampling=0, optimize=False, progressive=False)
    oversized = bytearray(black.getvalue())
    frame = oversized.index(b"\xff\xc0")
    oversized[frame + 5:frame + 9] = (65000).to_bytes(2, "big") * 2
    assert Path(generated["inputs"]["oversized"]["path"]).read_bytes() == oversized
    assert generated["inputs"]["small"]["bytes"] < 1048576 < generated["inputs"]["large"]["bytes"]
    assert generated["inputs"]["progressive"]["bytes"] > 1048576
    assert len(run.validate_stress(config, generated)) == 16


@pytest.mark.parametrize("attempt", ["attempt-01", "attempt-02"])
@pytest.mark.parametrize("failure, message", [("semantic-or-unclassified", "exhausted whole-epoch calibration"), ("external-interruption", "cancelled")])
def test_terminal_failure_cannot_resume_same_attempt(config, tmp_path, attempt, failure, message):
    control = tmp_path / ("control." + attempt)
    control.mkdir()
    (control / "completion.json").write_text(json.dumps({"success": False, "failure_kind": failure, "error": message, "elapsed_ns": 1000}))
    before = (control / "completion.json").read_bytes()
    with pytest.raises(ValueError, match="terminal failure"):
        run.check_retry(config, control, tmp_path / attempt)
    assert (control / "completion.json").read_bytes() == before


def test_resume_requires_clean_stage_boundary(config, tmp_path):
    control = tmp_path / "control.attempt-01"
    control.mkdir()
    run_dir = tmp_path / "attempt-01"
    assert run.check_retry(config, control, run_dir) == 0
    (control / "launch.json").write_text("{}")
    with pytest.raises(ValueError, match="interrupted"):
        run.check_retry(config, control, run_dir)
    (control / "completion.json").write_text(json.dumps({"success": False, "pending": ["validate", "timing", "memory", "native", "report"]}))
    (control / "stages.json").write_text(json.dumps({"preflight": {"exit_code": 0}}))
    assert run.check_retry(config, control, run_dir) == 0
    for exit_code in (None, 1):
        (control / "stages.json").write_text(json.dumps({"preflight": {"exit_code": 0}, "timing": {"exit_code": exit_code}}))
        with pytest.raises(ValueError, match="interrupted or failed stage"):
            run.check_retry(config, control, run_dir)


def test_supervisor_preserves_exhausted_calibration_failure(config, tmp_path, monkeypatch):
    """Exercise supervisor persistence without decoding or scientific timing."""
    settings = copy.deepcopy(config)
    settings["workflow_record_root"] = str(tmp_path)
    run_dir, control = tmp_path / "attempt-01", tmp_path / "control.attempt-01"
    run_dir.mkdir()
    control.mkdir()
    identity = {"code_sha": "reviewed-sha"}
    wheels = {name: {"sha256": name} for name in ("normal", "diagnostic")}
    binding = dict(identity, wheels={name: row["sha256"] for name, row in wheels.items()}, corpus_sha256=config["corpus"]["manifest_sha256"])
    (control / "stages.json").write_text(json.dumps({stage: {"exit_code": 0, "binding": binding, "artifacts": {}} for stage in ("preflight", "validate")}))
    monkeypatch.setattr(run, "code_identity", lambda _config: identity)
    monkeypatch.setattr(run, "run_paths", lambda *_args: (run_dir, control))
    monkeypatch.setattr(run, "service_envelope", lambda *_args: None)
    monkeypatch.setattr(run, "wheels", lambda *_args: wheels)
    monkeypatch.setattr(corpus, "select", lambda *_args: ([], b""))
    class Heartbeat:
        def __init__(self, *_args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def pulse(self):
            pass
    monkeypatch.setattr(run, "Heartbeat", Heartbeat)
    calls = []
    def exhausted(stage, *_args):
        assert json.loads((control / "stages.json").read_text())[stage]["exit_code"] is None
        calls.append(stage)
        raise ValueError("timing group exhausted duration calibration budget")
    monkeypatch.setattr(run, "run_stage", exhausted)
    with pytest.raises(ValueError, match="exhausted duration calibration"):
        run.supervise(settings, run_dir, "timing")
    completion = (control / "completion.json").read_bytes()
    stages = (control / "stages.json").read_bytes()
    assert json.loads(completion)["failure_kind"] == "semantic-or-unclassified"
    assert json.loads(stages)["timing"]["exit_code"] == 1
    with pytest.raises(ValueError, match="terminal failure"):
        run.supervise(settings, run_dir, "timing")
    assert calls == ["timing"]
    assert (control / "completion.json").read_bytes() == completion
    assert (control / "stages.json").read_bytes() == stages


def test_validation_digest_compares_worker_profiles(config):
    validation, _ = evidence(config)
    run.check_validation_digests(config, validation['cells'])
    for cell in run.configurations(config):
        if cell['workload'] == 'pipeline' and cell['workers'] == 2:
            validation['cells'][cell['config_id']]['result_digest'] = 'different-worker-result'
    with pytest.raises(ValueError, match='digests differ'):
        run.check_validation_digests(config, validation['cells'])


def test_low_memory_admission_is_explicit_and_preserves_other_checks(config, tmp_path, monkeypatch):
    original = copy.deepcopy(config)
    monkeypatch.setattr(run, "available_memory", lambda: 4 * 2**30)
    monkeypatch.setattr(run.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=20 * 2**30))
    monkeypatch.setattr(run.subprocess, "check_output", lambda *_args, **_kwargs: "1.5.0")
    monkeypatch.setattr(run.subprocess, "run", lambda *_args, **_kwargs: None)
    with pytest.raises(OSError, match="6 GiB"):
        run.resource_preflight(config, tmp_path)
    observed = run.resource_preflight(config, tmp_path, allow_low_memory=True)
    assert observed == {"available_bytes": 4 * 2**30, "required_bytes": 6 * 2**30,
                        "threshold_met": False, "allow_low_memory": True}
    assert config == original
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=2 * 2**30))
    with pytest.raises(OSError, match="free output storage"):
        run.resource_preflight(config, tmp_path, allow_low_memory=True)


def test_report_labels_and_validates_memory_admission(config):
    provenance = {"binding": {"allow_low_memory": True},
                  "resource_preflight": {"available_bytes": 4 * 2**30, "required_bytes": 6 * 2**30,
                                         "threshold_met": False, "allow_low_memory": True}}
    text = " ".join(report.memory_admission(config, provenance))
    assert "4.000 GiB" in text and "another machine" in text
    changed = copy.deepcopy(provenance)
    changed["binding"].clear()
    with pytest.raises(ValueError, match="memory admission"):
        report.memory_admission(config, changed)
    changed["resource_preflight"]["allow_low_memory"] = False
    with pytest.raises(ValueError, match="memory admission"):
        report.memory_admission(config, changed)
    changed["resource_preflight"].update(available_bytes=7 * 2**30, threshold_met=True)
    assert report.memory_admission(config, changed) == []


@pytest.mark.parametrize("allowed", [False, True])
def test_memory_admission_policy_cannot_change_on_resume(config, tmp_path, monkeypatch, allowed):
    settings = copy.deepcopy(config)
    settings["workflow_record_root"] = str(tmp_path)
    run_dir, control = tmp_path / "attempt-01", tmp_path / "control"
    monkeypatch.setattr(run, "code_identity", lambda _config: {"code_sha": "reviewed-sha"})
    monkeypatch.setattr(run, "run_paths", lambda *_args: (run_dir, control))
    monkeypatch.setattr(run, "wheels", lambda *_args: {name: {"sha256": name} for name in ("normal", "diagnostic")})
    monkeypatch.setattr(run.corpus, "select", lambda *_args: ([], b""))
    monkeypatch.setattr(run, "run_stage", lambda *_args: None)
    run.supervise(settings, run_dir, "preflight", allow_low_memory=allowed)
    before = (run_dir / "provenance.json").read_bytes()
    with pytest.raises(ValueError, match="attempt provenance changed"):
        run.supervise(settings, run_dir, "preflight", allow_low_memory=not allowed)
    assert (run_dir / "provenance.json").read_bytes() == before
