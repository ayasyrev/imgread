import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

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
                                row["output_nbytes"] = row["output_count"] * 256 * 256 * 3
                            rows.append(row)
    result = report.memory_evidence(config, rows, tmp_path)
    assert result["sequence_cases"] == 8 and result["manifest_cases"] > 0
    assert len(result["manifest_summary"]) > 0
    with pytest.raises(ValueError, match="incomplete"):
        report.memory_evidence(config, rows[:-1], tmp_path)


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
    from PIL import ImageFile
    settings = copy.deepcopy(config["sequence"])
    settings["large_shape"] = [768, 768, 3]
    previous = ImageFile.MAXBLOCK
    generated = corpus.generate(settings, tmp_path)
    assert ImageFile.MAXBLOCK == previous
    assert generated["inputs"]["progressive"]["bytes"] > 1048576
    assert len(run.validate_stress(config, generated)) == 10


def test_validation_digest_compares_worker_profiles(config):
    validation, _ = evidence(config)
    run.check_validation_digests(config, validation['cells'])
    for cell in run.configurations(config):
        if cell['workload'] == 'pipeline' and cell['workers'] == 2:
            validation['cells'][cell['config_id']]['result_digest'] = 'different-worker-result'
    with pytest.raises(ValueError, match='digests differ'):
        run.check_validation_digests(config, validation['cells'])
