"""Loader timing and memory study supervisor."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

import corpus

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("preflight", "validate", "timing", "memory", "native", "report")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_config(path):
    config = json.loads(Path(path).read_text())
    corpus_root = Path(config["corpus"]["root"]).expanduser()
    config["corpus"]["root"] = str((ROOT / corpus_root).resolve())
    validate_config(config)
    return config


def validate_config(config):
    for profile in config["pipeline"]["worker_profiles"]:
        pipeline_kwargs(config, profile["num_workers"])


def pipeline_kwargs(config, workers):
    expected = {0: {"num_workers": 0, "prefetch_factor": None, "persistent_workers": False, "multiprocessing_context": None},
                2: {"num_workers": 2, "prefetch_factor": 2, "persistent_workers": True, "multiprocessing_context": "spawn"}}
    profiles = config["pipeline"]["worker_profiles"]
    selected = [profile for profile in profiles if profile.get("num_workers") == workers]
    if workers not in expected or selected != [expected[workers]]:
        raise ValueError("invalid DataLoader worker profile")
    return dict(selected[0], batch_size=config["pipeline"]["batch_size"],
                pin_memory=config["pipeline"]["pin_memory"], drop_last=config["pipeline"]["drop_last"])


def configurations(config):
    return [dict(config_id=f"{workload}-{backend}-w{workers}-{variant}", workload=workload,
                 backend=backend, workers=workers, variant=variant)
            for workload in ("direct", "pipeline") for backend in config["backends"]
            for workers in ([0] if workload == "direct" else [0, 2]) for variant in config["variants"]]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def append_rows(path, rows):
    with Path(path).open("a", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_paths(run_dir):
    run_dir = Path(run_dir).resolve()
    return run_dir, run_dir / "control"


def memory_snapshot(pid):
    values = {"pid": pid, "rss_bytes": 0, "pss_bytes": 0, "peak_rss_bytes": 0}
    for filename, mapping in (("status", {"VmRSS": "rss_bytes", "VmHWM": "peak_rss_bytes"}),
                              ("smaps_rollup", {"Pss": "pss_bytes"})):
        with Path(f"/proc/{pid}/{filename}").open() as stream:
            for line in stream:
                key, _, rest = line.partition(":")
                if key in mapping:
                    values[mapping[key]] = int(rest.split()[0]) * 1024
    return values


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def resource_preflight(config, run_dir, allow_low_memory=False):
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("the study requires Linux x86_64")
    if not set(config["environment"]["cpus"]).issubset(os.sched_getaffinity(0)):
        raise ValueError("CPUs 0-3 unavailable")
    observed_memory = available_memory()
    required_memory = config["resources"]["min_available_bytes"]
    if observed_memory < required_memory and not allow_low_memory:
        raise OSError("study requires at least 6 GiB MemAvailable")
    if shutil.disk_usage(run_dir).free < config["resources"]["min_free_bytes"]:
        raise OSError("study requires at least 10 GiB free output storage")
    for executable in ("systemd-run", "systemctl", "heaptrack", "heaptrack_print"):
        if shutil.which(executable) is None:
            raise ValueError(f"external prerequisite missing: {executable}")
    for executable in ("heaptrack", "heaptrack_print"):
        version = subprocess.check_output([executable, "--version"], text=True, stderr=subprocess.STDOUT)
        if not re.search(r"\b1\.5\.0\b", version):
            raise ValueError(f"{executable} must be 1.5.0")
    subprocess.run(["systemctl", "--user", "show-environment"], check=True, stdout=subprocess.DEVNULL)
    return {"available_bytes": observed_memory, "required_bytes": required_memory,
            "threshold_met": observed_memory >= required_memory, "allow_low_memory": allow_low_memory}


def wheels():
    result = {}
    for name in ("normal", "diagnostic"):
        matches = list((ROOT / "target" / "loader-wheels" / name).glob("*.whl"))
        if len(matches) != 1:
            raise ValueError(f"exactly one {name} wheel is required")
        result[name] = {"path": str(matches[0])}
    if Path(result["normal"]["path"]).name != Path(result["diagnostic"]["path"]).name:
        raise ValueError("wheel version/ABI mismatch")
    return result


def installed_wheel(diagnostic):
    import imgread
    import imgread._native as native
    if native._debug_build:
        raise ValueError("measurements require a release build")
    if hasattr(imgread.Loader, "_debug_state") != diagnostic:
        raise ValueError("diagnostic/normal wheel mode mismatch")
    return str(native.__file__)


def install_wheel(wheel):
    subprocess.run(["uv", "pip", "install", "--python", str(ROOT / "benchmarks/loader/.venv/bin/python"),
                    "--offline", "--no-deps", "--force-reinstall", wheel["path"]], check=True)


def set_threads(config, use_torch=True):
    os.sched_setaffinity(0, config["environment"]["cpus"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if use_torch:
        worker_threads(0)


def worker_threads(_worker_id):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


class StudyDataset:
    def __init__(self, samples, variant, backend):
        import imgread
        start = time.perf_counter_ns()
        self.paths = [str(path) for path, _ in samples]
        self.labels = [label for _, label in samples]
        self.preparation_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        self.variant, self.backend = variant, backend
        self.function = imgread.load_numpy
        self.loader = None if variant == "function" else imgread.Loader(self.paths if variant == "index" else None, backend=backend)
        self.construction_ns = time.perf_counter_ns() - start

    def __len__(self):
        return len(self.paths)

    def image(self, index):
        if self.variant == "function":
            return self.function(self.paths[index], backend=self.backend)
        if self.variant == "path":
            return self.loader(self.paths[index])
        return self.loader[index]

    def __getitem__(self, index):
        import numpy as np
        import torch
        image = self.image(index)
        if min(image.shape[:2]) < 224:
            raise ValueError("corpus image is smaller than the crop")
        return torch.from_numpy(np.ascontiguousarray(image[:224, :224])), self.labels[index]


class WorkerSetup:
    def __init__(self, ready):
        self.ready = ready

    def __call__(self, _worker_id):
        worker_threads(_worker_id)
        self.ready.wait(timeout=5)


def make_work(config, cell, samples, order):
    dataset = StudyDataset(samples, cell["variant"], cell["backend"])
    start = time.perf_counter_ns()
    batches = None
    if cell["workload"] == "pipeline":
        from torch.utils.data import DataLoader
        ready = mp.get_context("spawn").Barrier(3) if cell["workers"] else None
        batches = DataLoader(dataset, sampler=order, worker_init_fn=WorkerSetup(ready) if ready else None,
                             **pipeline_kwargs(config, cell["workers"]))
        batches._study_ready = ready
    return dataset, batches, dataset.preparation_ns, dataset.construction_ns + time.perf_counter_ns() - start


def close_batches(batches):
    if batches is not None and batches._iterator is not None:
        batches._iterator._shutdown_workers()


def epoch(dataset, batches, order, validate=False):
    hasher = hashlib.sha256() if validate else None
    samples = 0
    if batches is not None and getattr(batches, "_study_ready", None) is not None and batches._iterator is None:
        iter(batches)
        batches._study_ready.wait(timeout=5)
    if batches is None:
        for index in order:
            array = dataset.image(index)
            if hasher is not None:
                hasher.update(canonical([array.shape, dataset.labels[index]]))
                hasher.update(array.tobytes())
            samples += 1
            del array
    else:
        for images, labels in batches:
            samples += len(labels)
            if hasher is not None:
                hasher.update(canonical(list(images.shape)))
                hasher.update(images.numpy().tobytes())
                hasher.update(labels.numpy().tobytes())
            del images, labels
    if samples != len(order):
        raise ValueError("sample count mismatch")
    return samples, hasher.hexdigest() if hasher is not None else None


def timing_job(config, cell, samples, order, repetitions, validate=False):
    dataset, batches, preparation, construction = make_work(config, cell, samples, order)
    try:
        # Worker startup is separate from the complete untimed warmup epoch.
        start = time.perf_counter_ns()
        if batches is not None and cell["workers"]:
            iter(batches)
            batches._study_ready.wait(timeout=5)
            startup = time.perf_counter_ns() - start
        else:
            startup = 0
        warm_start = time.perf_counter_ns()
        count, result_digest = epoch(dataset, batches, order, validate)
        warm_ns = time.perf_counter_ns() - warm_start
        rows = []
        for number in range(repetitions):
            start = time.perf_counter_ns()
            count, _ = epoch(dataset, batches, order)
            elapsed = time.perf_counter_ns() - start
            rows.append({"epoch": number, "samples": count, "elapsed_ns": elapsed,
                         "startup_ns": startup + construction, "preparation_ns": preparation})
        return {"rows": rows, "warm_ns": warm_ns, "result_digest": result_digest,
                "startup_ns": startup + construction, "preparation_ns": preparation}
    finally:
        close_batches(batches)


def process_group(command, stdout, stderr, timeout, heartbeat=None):
    """No surviving workers after timeout, cancellation, or an exception."""
    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
    start = time.monotonic()
    try:
        while process.poll() is None:
            if time.monotonic() - start >= timeout:
                raise TimeoutError(f"child exceeded {timeout}s")
            if heartbeat:
                heartbeat()
            time.sleep(0.1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        return process.returncode
    finally:
        # The process group can outlive its leader (multiprocessing helpers).
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def child_job(config, job, run_dir, name, heartbeat=None):
    jobs = run_dir / "jobs"
    jobs.mkdir(exist_ok=True, mode=0o700)
    request, result = jobs / f"{name}.request.json", jobs / f"{name}.result.json"
    resume = 0
    while request.exists() or result.exists():
        resume += 1
        request, result = jobs / f"{name}-resume{resume}.request.json", jobs / f"{name}-resume{resume}.result.json"
    name = request.name.removesuffix(".request.json")
    atomic_json(request, dict(job, result_path=str(result)))
    with (jobs / f"{name}.stdout").open("w") as out, (jobs / f"{name}.stderr").open("w") as err:
        process_group([sys.executable, str(Path(__file__).resolve()), "--config", str(run_dir / "study.json"), "--run-dir", str(run_dir), "--worker-request", str(request)], out, err, config["resources"]["config_seconds"], heartbeat)
    return json.loads(result.read_text())


def zero_state(loader):
    return loader._debug_state() if loader is not None else {"pid": os.getpid(), "input_len": 0, "input_capacity": 0, "native_live": 0, "native_creations": 0, "manifest_bytes": 0, "manifest_entries": 0}


def manifest_receive(connection, deadline, message):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not connection.poll(remaining):
        raise TimeoutError(message)
    return connection.recv()


def manifest_child(paths, loader, connection, deadline):
    try:
        connection.send({"ready": os.getpid(), "state": zero_state(loader)})
        for phase in range(2):
            if manifest_receive(connection, deadline, "manifest phase handshake") != phase:
                raise ValueError("manifest phase sequence changed")
            connection.send(dict(memory_snapshot(os.getpid()), diagnostics=zero_state(loader), phase=phase, retained_python_paths=len(paths)))
    finally:
        connection.close()


def manifest_job(config, job):
    import gc
    import imgread
    import pickle
    # The first ready child must survive the other child's spawn/unpickle cost.
    # All startup and phase waits share this deadline; the outer process group
    # additionally enforces config_seconds from before worker interpreter startup.
    deadline = time.monotonic() + config["resources"]["config_seconds"]
    start = time.perf_counter_ns()
    paths = corpus.manifest_paths(job["length"])
    list_ns = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    loader = None if job["variant"] == "function" else imgread.Loader(paths if job["variant"] == "index" else None)
    construct_ns = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    payload = pickle.dumps((paths, loader), protocol=pickle.HIGHEST_PROTOCOL)
    pickle_ns = time.perf_counter_ns() - start
    payload_size = len(payload)
    del payload
    gc.collect()
    processes, pipes, rows = [], [], []
    start = time.perf_counter_ns()
    try:
        if job["workers"]:
            ctx = mp.get_context(job["method"])
            for _ in range(2):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=manifest_child, args=(paths, loader, child, deadline))
                processes.append(process)
                pipes.append(parent)
                process.start()
                child.close()
            for pipe in pipes:
                manifest_receive(pipe, deadline, "manifest worker startup")
        startup_ns = time.perf_counter_ns() - start
        for phase in range(2):
            for pipe in pipes:
                pipe.send(phase)
            for pipe in pipes:
                rows.append(dict(manifest_receive(pipe, deadline, "manifest worker phase"), role="worker"))
            rows.append(dict(memory_snapshot(os.getpid()), phase=phase, role="parent", diagnostics=zero_state(loader), retained_python_paths=len(paths)))
        return {"rows": rows, "list_ns": list_ns, "construction_ns": construct_ns, "pickle_bytes": payload_size, "pickle_ns": pickle_ns, "startup_ns": startup_ns}
    finally:
        pending_error = sys.exc_info()[0] is not None
        failures = []
        for pipe in pipes:
            pipe.close()
        for process in processes:
            if process.pid:
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                if process.exitcode != 0:
                    failures.append(process.exitcode)
        if failures and not pending_error:
            raise RuntimeError(f"manifest workers failed: {failures}")


def checked_image(loader, path, kind):
    expected = {"corrupt": RuntimeError, "oversized": ValueError}.get(kind)
    try:
        image = loader(path)
    except Exception as error:
        if expected is None or not isinstance(error, expected):
            raise
        return None
    if expected is not None:
        raise ValueError(f"stress fixture unexpectedly accepted: {kind}")
    return image


def sequence_phases(loader, config, inputs, epoch_number):
    import gc
    for name, kind, count in config["sequence"]["phases"]:
        rows = []
        for call in range(count):
            image = checked_image(loader, inputs[kind]["path"], kind)
            del image
            diagnostics = loader._debug_state()
            if diagnostics["input_len"] or diagnostics["input_capacity"] > config["active_cap"] or diagnostics["native_live"] > 1:
                raise ValueError("input/native retention violation")
            if kind == "small" and config["active_backend"] == "auto" and diagnostics["native_live"] != 1:
                raise ValueError("missing expected native reuse")
            rows.append({"call": call, "diagnostics": diagnostics})
        yield {"phase": name, "epoch": epoch_number, "pid": os.getpid(), "calls": rows, "output_nbytes": 0}
    if epoch_number == 1:
        arrays = [loader(inputs["small"]["path"]) for _ in range(config["sequence"]["retained_outputs"])]
        yield {"phase": "outputs-retained", "epoch": epoch_number, "pid": os.getpid(), "diagnostics": loader._debug_state(), "output_nbytes": sum(array.nbytes for array in arrays), "output_count": len(arrays)}
        del arrays
        gc.collect()
        yield {"phase": "outputs-released", "epoch": epoch_number, "pid": os.getpid(), "diagnostics": loader._debug_state(), "output_nbytes": 0, "output_count": 0}


class SequenceSource:
    """Mixed into torch IterableDataset lazily, keeping config-only imports light."""
    def __init__(self, config, inputs, barrier):
        self.config, self.inputs, self.barrier = config, inputs, barrier
        self.loader = None
        self.epoch = 0

    def __iter__(self):
        import gc
        import imgread
        if self.loader is None:
            self.loader = imgread.Loader(backend=self.config["active_backend"], max_buffer_bytes=self.config["active_cap"])
        for record in sequence_phases(self.loader, self.config, self.inputs, self.epoch):
            yield record
            if self.barrier is not None:
                self.barrier.wait(timeout=5)
        if self.epoch == 1:
            self.loader = None
            gc.collect()
            yield {"phase": "destroyed", "epoch": 1, "pid": os.getpid(), "diagnostics": zero_state(None), "output_nbytes": 0}
            if self.barrier is not None:
                self.barrier.wait(timeout=5)
        self.epoch += 1


def sequence_dataset_type():
    # The returned class lives in an importable companion module for spawn.
    from datasets import SequenceDataset
    return SequenceDataset


def sequence_job(config, job, inputs):
    from torch.utils.data import DataLoader
    active = dict(config, active_cap=job["cap"], active_backend=job["backend"])
    barrier = mp.get_context("spawn").Barrier(3) if job["workers"] else None
    dataset = sequence_dataset_type()(active, inputs, barrier)
    batches = DataLoader(dataset, batch_size=None, num_workers=job["workers"],
                         prefetch_factor=2 if job["workers"] else None,
                         persistent_workers=bool(job["workers"]),
                         multiprocessing_context="spawn" if job["workers"] else None,
                         pin_memory=False, worker_init_fn=worker_threads)
    rows, worker_pids = [], None
    try:
        for epoch_number in range(2):
            pending = []
            for record in batches:
                pending.append(record)
                if len(pending) == max(1, job["workers"]):
                    if len({record["phase"] for record in pending}) != 1:
                        raise ValueError("sequence workers crossed phase barrier")
                    for item in pending:
                        rows.append(dict(item, **{key: value for key, value in memory_snapshot(item["pid"]).items() if key != "pid"}, role="worker" if job["workers"] else "parent"))
                    if job["workers"]:
                        current = sorted(record["pid"] for record in pending)
                        if worker_pids is not None and current != worker_pids:
                            raise ValueError("persistent worker PID changed")
                        worker_pids = current
                        rows.append(dict(memory_snapshot(os.getpid()), epoch=epoch_number, phase=pending[0]["phase"], role="parent", output_nbytes=0))
                        barrier.wait(timeout=5)
                    pending.clear()
            if pending:
                raise ValueError("incomplete worker phase")
        return {"rows": rows}
    finally:
        if barrier:
            barrier.abort()
        close_batches(batches)


class MemorySampler:
    def __init__(self, path, interval):
        self.path, self.interval = path, interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.errors = []

    def run(self):
        try:
            while not self.stop.is_set():
                pids = {os.getpid()}
                frontier = list(pids)
                while frontier:
                    pid = frontier.pop()
                    try:
                        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
                    except FileNotFoundError:
                        continue
                    for value in children:
                        child = int(value)
                        if child not in pids:
                            pids.add(child)
                            frontier.append(child)
                rows = []
                sample_ns = time.perf_counter_ns()
                for pid in sorted(pids):
                    try:
                        rows.append(dict(memory_snapshot(pid), sample_ns=sample_ns, role="parent" if pid == os.getpid() else "child"))
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                append_rows(self.path, rows)
                self.stop.wait(self.interval)
        except BaseException as error:
            self.errors.append(error)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(5)
        if self.thread.is_alive() or self.errors:
            raise RuntimeError(f"memory sampler failed: {self.errors}")


def preflight_job(config):
    import torch
    if platform.python_version() != config["environment"]["python"]:
        raise ValueError("Python version differs from locked study")
    for name in ("numpy", "pillow", "pytest", "torch", "torchvision"):
        if importlib.metadata.version(name) != config["environment"][name]:
            raise ValueError(f"dependency version mismatch: {name}")
    package_origin = installed_wheel(False)
    from torch.utils.data import DataLoader
    constructed = []
    for cell in configurations(config):
        if cell["workload"] == "pipeline":
            dataset, loader, _, _ = make_work(config, cell, [], [])
            kwargs = pipeline_kwargs(config, cell["workers"])
            constructed.append({"config_id": cell["config_id"], "kwargs": kwargs})
            del loader, dataset
    if len(constructed) != 12:
        raise ValueError("incomplete DataLoader preflight")
    versions = {}
    for name, command in {"uv": ["uv", "--version"], "rustc": ["rustc", "-Vv"], "cargo": ["cargo", "-V"],
                          "cc": ["cc", "--version"], "heaptrack": ["heaptrack", "--version"], "heaptrack_print": ["heaptrack_print", "--version"]}.items():
        versions[name] = {"path": shutil.which(command[0]), "version": subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)}
    return {"pipeline_constructors": constructed, "installed_extension": package_origin, "tools": versions,
            "python": sys.version, "platform": platform.platform(), "libc": platform.libc_ver(),
            "cpuinfo": Path("/proc/cpuinfo").read_text(), "affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "environment": {name: os.environ.get(name) for name in ("CARGO_PROFILE_RELEASE_DEBUG", "CARGO_PROFILE_RELEASE_STRIP", "CFLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")}}


def validate_stress(config, generated):
    import imgread
    import warnings
    rows = []
    if set(generated["inputs"]) != set(corpus.SYNTHETIC_FILES):
        raise ValueError("incomplete synthetic input matrix")
    for backend in config["backends"]:
        loader = imgread.Loader(backend=backend)
        for kind, item in generated["inputs"].items():
            path = item["path"]
            def capture(call):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        image = call(path)
                        if kind in ("corrupt", "oversized"):
                            raise AssertionError(f"deterministic {kind} fixture was accepted by {backend}")
                        result = {"shape": image.shape, "sha256": hashlib.sha256(image.tobytes()).hexdigest()}
                    except (ValueError, RuntimeError) as error:
                        if kind not in ("corrupt", "oversized"):
                            raise
                        result = {"error_type": type(error).__name__, "error": str(error)}
                    return dict(result, warnings=[str(w.message) for w in caught])
            expected = capture(lambda path: imgread.load_numpy(path, backend=backend))
            if capture(loader) != expected:
                raise ValueError("synthetic Loader/function parity mismatch")
            rows.append({"backend": backend, "kind": kind, "result": expected})
    return rows


def validate_request(config, request, run_dir):
    kind = request.get("kind")
    keys = {"kind", "result_path"}
    if kind in ("timing", "validate"):
        keys |= {"cell", "repetitions"}
        if request.get("cell") not in configurations(config):
            raise ValueError("worker cell is outside the matrix")
        if type(request.get("repetitions", 0)) is not int or request.get("repetitions", 0) < 0:
            raise ValueError("invalid calibrated repetition count")
        if kind == "validate" and request.get("repetitions", 0) != 0:
            raise ValueError("validation cannot emit timing epochs")
    elif kind == "manifest":
        keys |= {"length", "workers", "method", "variant", "repeat"}
        if request.get("length") not in config["manifest"]["lengths"] or request.get("workers") not in (0, 2) or request.get("variant") not in config["variants"] or request.get("repeat") not in range(3):
            raise ValueError("manifest job is outside the matrix")
        methods = [method for method in config["manifest"]["start_methods"] if method in mp.get_all_start_methods()] if request["workers"] else [None]
        if request.get("method") not in methods:
            raise ValueError("manifest process method mismatch")
    elif kind == "sequence":
        keys |= {"backend", "cap", "workers", "method"}
        if request.get("backend") not in config["backends"] or request.get("cap") not in config["sequence"]["caps"] or request.get("workers") not in (0, 2) or request.get("method") != ("spawn" if request["workers"] else None):
            raise ValueError("sequence job is outside the matrix")
    elif kind not in ("preflight", "stress"):
        raise ValueError("unknown internal worker request")
    if set(request) - keys or "result_path" not in request:
        raise ValueError("unexpected internal worker request fields")
    result = Path(request["result_path"]).resolve()
    if result.parent != run_dir / "jobs" or not result.name.endswith(".result.json") or result.exists():
        raise ValueError("worker results must be new files in this attempt's jobs directory")


def worker(config, request, run_dir):
    validate_request(config, request, run_dir)
    if request["kind"] != "preflight":
        service_envelope(config)
    stage = request["kind"]
    set_threads(config, use_torch=stage != "manifest")
    diagnostic = stage in ("manifest", "sequence")
    installed_wheel(diagnostic)
    if stage == "preflight":
        result = preflight_job(config)
    elif stage in ("timing", "validate"):
        samples = corpus.samples_from_manifest(config["corpus"], run_dir / "corpus.tsv")
        order = list(range(len(samples)))
        random.Random(config["timing"]["order_seed"]).shuffle(order)
        result = timing_job(config, request["cell"], samples, order, request.get("repetitions", 0), stage == "validate")
    elif stage == "stress":
        result = validate_stress(config, json.loads((run_dir / "generated-inputs.json").read_text()))
    else:
        with MemorySampler(run_dir / "jobs" / (Path(request["result_path"]).stem + ".samples.jsonl"), config["resources"]["sample_seconds"]):
            if stage == "manifest":
                result = manifest_job(config, request)
            elif stage == "sequence":
                generated = json.loads((run_dir / "generated-inputs.json").read_text())
                result = sequence_job(config, request, generated["inputs"])
            else:
                raise ValueError("unknown worker job")
    atomic_json(request["result_path"], result)


def cohort(cell):
    return cell["workload"], cell["backend"], cell["workers"]


def check_validation_digests(config, cells):
    matrix = configurations(config)
    for workload, backend in {(cell["workload"], cell["backend"]) for cell in matrix}:
        digests = {cells[cell["config_id"]]["result_digest"] for cell in matrix
                   if cell["workload"] == workload and cell["backend"] == backend}
        if len(digests) != 1 or None in digests:
            raise ValueError("variant/worker output pixel and label digests differ")


def timing_stage(config, run_dir, heartbeat):
    validation = json.loads((run_dir / "validation.json").read_text())
    matrix = configurations(config)
    groups = sorted({cohort(cell) for cell in matrix})
    for round_number in range(config["timing"]["rounds"]):
        randomizer = random.Random(config["timing"]["round_seed"] + round_number)
        shuffled = matrix.copy()
        randomizer.shuffle(shuffled)
        repetitions = {}
        for group in groups:
            calibrations = []
            for cell in (cell for cell in shuffled if cohort(cell) == group):
                result = child_job(config, {"kind": "timing", "cell": cell}, run_dir,
                                   f"calibration-r{round_number}-{cell['config_id']}", heartbeat)
                calibrations.append(result["warm_ns"] / 1e9)
            repetitions[group] = max(2, math.ceil(config["timing"]["calibration_target_seconds"] / min(calibrations)))
        results = {}
        for cell in shuffled:
            result = child_job(config, {"kind": "timing", "cell": cell, "repetitions": repetitions[cohort(cell)]}, run_dir,
                               f"timing-r{round_number}-a0-{cell['config_id']}", heartbeat)
            results[cell["config_id"]] = result
        for group in groups:
            cells = [cell for cell in shuffled if cohort(cell) == group]
            for attempt in range(config["timing"]["max_calibrations"]):
                valid = all(sum(row["elapsed_ns"] for row in results[cell["config_id"]]["rows"]) >= 2_000_000_000 for cell in cells)
                for cell in cells:
                    rows = [dict(row, **cell, round=round_number, calibration=attempt,
                                 order_digest=validation["order_digest"], result_digest=validation["cells"][cell["config_id"]]["result_digest"],
                                 binding=validation["binding"])
                            for row in results[cell["config_id"]]["rows"]]
                    append_rows(run_dir / ("timings.jsonl" if valid else "timings-discarded.jsonl"), rows)
                if valid:
                    break
                if attempt + 1 == config["timing"]["max_calibrations"]:
                    raise ValueError("timing group exhausted duration calibration budget")
                repetitions[group] *= 2
                randomizer.shuffle(cells)
                for cell in cells:
                    results[cell["config_id"]] = child_job(config, {"kind": "timing", "cell": cell, "repetitions": repetitions[group]}, run_dir,
                                                           f"timing-r{round_number}-a{attempt+1}-{cell['config_id']}", heartbeat)


def memory_stage(config, run_dir, heartbeat):
    for length in config["manifest"]["lengths"]:
        for workers in config["manifest"]["workers"]:
            methods = [method for method in config["manifest"]["start_methods"] if method in mp.get_all_start_methods()] if workers else [None]
            for method in methods:
                for variant in config["variants"]:
                    for repeat in range(3):
                        job = dict(kind="manifest", length=length, workers=workers, method=method, variant=variant, repeat=repeat)
                        name = f"manifest-n{length}-w{workers}-{method}-{variant}-r{repeat}"
                        result = child_job(config, job, run_dir, name, heartbeat)
                        append_rows(run_dir / "memory.jsonl", [dict(row, **job, job_id=name, preparation={key: value for key, value in result.items() if key != "rows"}) for row in result["rows"]])
    for backend in config["backends"]:
        for cap in config["sequence"]["caps"]:
            for workers in config["sequence"]["workers"]:
                job = dict(kind="sequence", backend=backend, cap=cap, workers=workers, method="spawn" if workers else None)
                name = f"sequence-{backend}-cap{cap}-w{workers}"
                result = child_job(config, job, run_dir, name, heartbeat)
                append_rows(run_dir / "memory.jsonl", [dict(row, **job, job_id=name) for row in result["rows"]])


def native_stage(config, run_dir, heartbeat):
    directory = run_dir / "native"
    directory.mkdir(mode=0o700, exist_ok=True)
    manifest = []
    for checkpoint in config["native"]["checkpoints"]:
        for repeat in range(1, 4):
            stem = directory / f"{checkpoint}-{repeat:02d}"
            command = ["uv", "run", "--project", "benchmarks/loader", "--no-sync", "heaptrack", "--record-only", "-o", str(stem),
                       str(ROOT / "benchmarks/loader/.venv/bin/python"), str(ROOT / "benchmarks/loader/native_probe.py"),
                       "--run-dir", str(run_dir), "--checkpoint", checkpoint, "--repeat", str(repeat)]
            with stem.with_suffix(".stdout").open("w") as out, stem.with_suffix(".stderr").open("w") as err:
                process_group(command, out, err, config["resources"]["native_seconds"], heartbeat)
            profiles = [path for path in directory.glob(stem.name + "*") if path.suffix in (".gz", ".zst")]
            if len(profiles) != 1:
                raise ValueError("expected exactly one heaptrack compressed profile")
            profile = profiles[0]
            exports = {}
            for cost, label in (("leaked", "live"), ("peak", "peak")):
                stacks = Path(str(stem) + f".{label}.stacks")
                command = ["uv", "run", "--project", "benchmarks/loader", "--no-sync", "heaptrack_print", str(profile)]
                if cost == "leaked":
                    command.extend(["--print-leaks", "--disable-builtin-suppressions", "--disable-embedded-suppressions"])
                command.extend(["--flamegraph-cost-type", cost, "--print-flamegraph", str(stacks)])
                with Path(str(stem) + f".{label}.txt").open("w") as out, Path(str(stem) + f".{label}.stderr").open("w") as err:
                    process_group(command, out, err, config["resources"]["native_seconds"], heartbeat)
                exports[label] = {"path": str(stacks)}
            manifest.append({"checkpoint": checkpoint, "repeat": repeat, "profile": str(profile),
                             "compression": profile.suffix, "exports": exports})
    atomic_json(run_dir / "native-profiles.json", manifest)


class Heartbeat:
    def __init__(self, config, control, run_dir, start_ns):
        self.config, self.control, self.run_dir, self.start_ns = config, control, run_dir, start_ns
        self.stage = "setup"
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def pulse(self):
        if self.error:
            raise self.error
        if (time.monotonic_ns() - self.start_ns) / 1e9 > self.config["resources"]["attempt_seconds"]:
            raise TimeoutError("attempt exceeded its one-hour budget")
        if shutil.disk_usage(self.run_dir).free < self.config["resources"]["stop_free_bytes"]:
            raise OSError("free artifact storage below 2 GiB")

    def loop(self):
        try:
            while not self.stop.is_set():
                self.pulse()
                atomic_json(self.control / "heartbeat.json", {"stage": self.stage, "pid": os.getpid(), "wall_ns": time.time_ns(), "elapsed_ns": time.monotonic_ns() - self.start_ns})
                self.stop.wait(self.config["resources"]["heartbeat_seconds"])
        except BaseException as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(5)


def verify_stage(record, run_dir, binding):
    if record.get("exit_code") != 0 or record.get("binding") != binding:
        raise ValueError("stage is incomplete or belongs to another run")
    for relative in record["artifacts"]:
        path = (run_dir / relative).resolve(strict=True)
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise ValueError("completed stage artifact missing or outside run directory")


def check_resume(control):
    current_path = control / "completion.json"
    if current_path.exists():
        current = json.loads(current_path.read_text())
        if "failure_kind" in current or "error" in current:
            raise ValueError("run has a terminal failure; use a new run directory")
    elif (control / "launch.json").exists():
        raise ValueError("run was interrupted without a completion record; use a new run directory")
    stage_path = control / "stages.json"
    if stage_path.exists():
        stages = json.loads(stage_path.read_text())
        if any(record.get("exit_code") != 0 for record in stages.values()):
            raise ValueError("run has an interrupted or failed stage; use a new run directory")


def run_stage(stage, config, run_dir, provenance, heartbeat):
    if stage == "preflight":
        provenance["resource_preflight"] = resource_preflight(config, run_dir, provenance["binding"].get("allow_low_memory", False))
        atomic_json(run_dir / "provenance.json", provenance)
        samples, manifest = corpus.select(config["corpus"])
        from PIL import Image
        for path, _ in samples:
            with Image.open(path) as image:
                if min(image.size) < config["corpus"]["min_dimension"]:
                    raise ValueError("natural corpus violates crop dimensions")
        (run_dir / "corpus.tsv").write_bytes(manifest)
        generated = corpus.generate(config["sequence"], run_dir / "generated")
        atomic_json(run_dir / "generated-inputs.json", generated)
        data = child_job(config, {"kind": "preflight"}, run_dir, "preflight", heartbeat)
        provenance.update(data)
        atomic_json(run_dir / "provenance.json", provenance)
    elif stage == "validate":
        samples = corpus.samples_from_manifest(config["corpus"], run_dir / "corpus.tsv")
        order = list(range(len(samples)))
        random.Random(config["timing"]["order_seed"]).shuffle(order)
        cells = {}
        for cell in configurations(config):
            cells[cell["config_id"]] = child_job(config, {"kind": "validate", "cell": cell}, run_dir, "validate-" + cell["config_id"], heartbeat)
        check_validation_digests(config, cells)
        stress = child_job(config, {"kind": "stress"}, run_dir, "stress", heartbeat)
        atomic_json(run_dir / "validation.json", {"binding": provenance["binding"], "order_digest": hashlib.sha256(canonical(order)).hexdigest(), "cells": cells, "stress": stress})
    elif stage == "timing":
        timing_stage(config, run_dir, heartbeat)
    elif stage == "memory":
        memory_stage(config, run_dir, heartbeat)
    elif stage == "native":
        native_stage(config, run_dir, heartbeat)
    elif stage == "report":
        import report
        result = report.generate(config, run_dir, check_complete=True)
        if result["decision"] != "completed":
            raise ValueError("material regression: measured report requires owner decision")


def classify_failure(error):
    if isinstance(error, KeyboardInterrupt):
        return "external-interruption"
    if isinstance(error, (MemoryError, OSError, TimeoutError)):
        return "resource-exhaustion"
    return "semantic-or-unclassified"



def service_envelope(config):
    unit = os.environ.get("IMGREAD_UNIT", "imgread-loader-study")
    output = subprocess.check_output(["systemctl", "--user", "show", unit,
                                      "--property=ControlGroup,MemoryMax,TasksMax,KillMode,RuntimeMaxUSec,TimeoutStopUSec"], text=True, timeout=10)
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    cgroup = fields.get("ControlGroup", "")
    if not cgroup or not any(line.endswith(":" + cgroup) for line in Path("/proc/self/cgroup").read_text().splitlines()):
        raise ValueError("measurements must run in the named systemd study service")
    expected = {"MemoryMax": str(config["resources"]["memory_max_bytes"]), "TasksMax": "64", "KillMode": "control-group", "RuntimeMaxUSec": "1h", "TimeoutStopUSec": "10s"}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError(f"systemd resource envelope differs from contract: {fields}")
    return fields


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file
    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.flush()
        return len(text)
    def flush(self):
        self.console.flush()
        self.file.flush()


def supervise(config, run_dir, selected_stage, allow_low_memory=False):
    run_dir, control = run_paths(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    control.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = ROOT / "target" / "loader-study.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_resume(control)
        if selected_stage != "preflight":
            service_envelope(config)
        wheel_info = wheels()
        binding = {"run_dir": str(run_dir)}
        if allow_low_memory:
            binding["allow_low_memory"] = True
        provenance_path = run_dir / "provenance.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if provenance["binding"] != binding or load_config(run_dir / "study.json") != config:
                raise ValueError("attempt provenance changed")
        else:
            provenance = dict(binding=binding, wheels=wheel_info, command=sys.argv, created_ns=time.time_ns())
            atomic_json(provenance_path, provenance)
            atomic_json(run_dir / "study.json", config)
        launch_path = control / "launch.json"
        if launch_path.exists():
            launch = json.loads(launch_path.read_text())
            if launch.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
                raise ValueError("cannot resume a budget across host reboot; record an external interruption")
            if launch["binding"] != binding:
                raise ValueError("attempt ledger binding changed")
        else:
            launch = {"binding": binding, "started_ns": time.monotonic_ns(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "attempt": run_dir.name, "command": sys.argv}
            atomic_json(launch_path, launch)
        stage_path = control / "stages.json"
        records = json.loads(stage_path.read_text()) if stage_path.exists() else {}
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("study cancelled")
        previous_signal = signal.signal(signal.SIGTERM, interrupted)
        installed_mode = None
        success = False
        active_stage = None
        try:
            with Heartbeat(config, control, run_dir, launch["started_ns"]) as heartbeat:
                for stage in STAGES:
                    if selected_stage != "all" and stage != selected_stage:
                        continue
                    active_stage = heartbeat.stage = stage
                    heartbeat.pulse()
                    for predecessor in STAGES[:STAGES.index(stage)]:
                        if predecessor not in records:
                            raise ValueError(f"stage {stage} requires complete {predecessor}")
                        verify_stage(records[predecessor], run_dir, binding)
                    if stage in records and records[stage].get("exit_code") == 0:
                        verify_stage(records[stage], run_dir, binding)
                        continue
                    # Persist admission before any stage work. An uncatchable
                    # interruption cannot masquerade as a clean stage boundary.
                    records[stage] = {"exit_code": None, "binding": binding, "started_ns": time.monotonic_ns()}
                    atomic_json(stage_path, records)
                    for filename in ({"timing": ["timings.jsonl", "timings-discarded.jsonl"], "memory": ["memory.jsonl"], "native": ["native-profiles.json"]}.get(stage, [])):
                        path = run_dir / filename
                        if path.exists():
                            path.rename(path.with_name(path.name + f".incomplete-{time.time_ns()}"))
                    if stage != "report":
                        mode = "diagnostic" if stage in ("memory", "native") else "normal"
                        if mode != installed_mode:
                            install_wheel(wheel_info[mode])
                            installed_mode = mode
                    if stage == "native" and (run_dir / "native").exists():
                        (run_dir / "native").rename(run_dir / f"native.incomplete-{time.time_ns()}")
                    start = time.perf_counter_ns()
                    print(f"Starting {stage}", flush=True)
                    run_stage(stage, config, run_dir, provenance, heartbeat.pulse)
                    artifacts = [str(path.relative_to(run_dir)) for path in sorted(run_dir.rglob("*"))
                                 if path.is_file() and not path.is_relative_to(control)
                                 and path.name not in ("stdout.log", "stderr.log")]
                    records[stage] = {"exit_code": 0, "elapsed_ns": time.perf_counter_ns() - start, "binding": binding, "artifacts": artifacts}
                    atomic_json(stage_path, records)
                success = all(stage in records and records[stage]["exit_code"] == 0 for stage in STAGES)
                atomic_json(control / "completion.json", {"success": success, "binding": binding, "pending": [stage for stage in STAGES if stage not in records], "elapsed_ns": time.monotonic_ns() - launch["started_ns"]})
        except BaseException as error:
            if active_stage and records.get(active_stage, {}).get("exit_code") != 0:
                records[active_stage] = {"exit_code": 1, "binding": binding, "error": repr(error)}
                atomic_json(stage_path, records)
            atomic_json(control / "completion.json", {"success": False, "binding": binding, "failure_kind": classify_failure(error), "error": repr(error), "elapsed_ns": time.monotonic_ns() - launch["started_ns"]})
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_signal)
            if installed_mode == "diagnostic":
                install_wheel(wheel_info["normal"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="study settings; defaults to saved run settings when resuming")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--corpus-root", type=Path, help="image corpus directory; overrides the config")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="preflight")
    parser.add_argument("--allow-low-memory", action="store_true", help="allow less than 6 GiB available memory; record this limitation")
    parser.add_argument("--self-check", action="store_true", help="validate settings without measurements")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    saved_config = args.run_dir / "study.json" if args.run_dir else None
    config_path = args.config or (saved_config if saved_config and saved_config.is_file()
                                  else ROOT / "benchmarks/loader/study.json")
    config = load_config(config_path)
    if args.corpus_root is not None:
        config["corpus"]["root"] = str(args.corpus_root.expanduser().resolve())
    if args.self_check:
        print(json.dumps({"configurations": len(configurations(config))}))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required for study stages")
    if args.worker_request:
        request = args.worker_request.resolve(strict=True)
        if not request.is_relative_to(args.run_dir.resolve() / "jobs"):
            raise ValueError("worker request must be inside this attempt's jobs directory")
        worker(config, json.loads(request.read_text()), args.run_dir.resolve())
    else:
        run_dir, _control = run_paths(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (run_dir / "stdout.log").open("a") as out, (run_dir / "stderr.log").open("a") as err:
            with contextlib.redirect_stdout(Tee(sys.stdout, out)), contextlib.redirect_stderr(Tee(sys.stderr, err)):
                try:
                    supervise(config, run_dir, args.stage, args.allow_low_memory)
                except BaseException:
                    traceback.print_exc()
                    raise SystemExit(1)


if __name__ == "__main__":
    main()
