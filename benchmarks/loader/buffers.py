"""Exploratory function-versus-Loader buffer benchmark, separate from the path/index matrix."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import imgread
import numpy as np
import torch
from torch.utils.data import DataLoader

from corpus import select

class BufferWork:
    def __init__(self, encoded, labels, variant, backend, representation):
        self.encoded, self.labels = encoded, labels
        self.backend, self.representation = backend, representation
        self.loader = imgread.Loader(backend=backend) if variant == "loader" else None

    def __len__(self):
        return len(self.encoded)

    def image(self, index):
        data = self.encoded[index]
        if self.representation == "memoryview":
            data = memoryview(data)
        if self.loader is None:
            return imgread.load_numpy_from_bytes(data, backend=self.backend)
        return self.loader.decode(data)

    def __getitem__(self, index):
        array = self.image(index)
        if min(array.shape[:2]) < 224:
            raise ValueError("input smaller than 224 crop")
        return torch.from_numpy(np.ascontiguousarray(array[:224, :224])), self.labels[index]


def worker_init(_index):
    torch.set_num_threads(1)


def epoch(dataset, batches, order, validate=False):
    digest = hashlib.sha256() if validate else None
    count = 0
    if batches is None:
        for index in order:
            array = dataset.image(index)
            if digest is not None:
                digest.update(json.dumps([list(array.shape), dataset.labels[index]]).encode())
                digest.update(array.tobytes())
            count += 1
    else:
        for images, labels in batches:
            count += len(labels)
            if digest is not None:
                digest.update(images.numpy().tobytes())
                digest.update(labels.numpy().tobytes())
    if count != len(order):
        raise RuntimeError("sample count mismatch")
    return digest.hexdigest() if digest is not None else None


def close(batches):
    if batches is not None and batches._iterator is not None:
        batches._iterator._shutdown_workers()


def cohort(encoded, labels, order, backend, representation, workers, args, rng):
    variants = ["function", "loader"]
    work = {}
    metadata = {}
    try:
        # Only these two variants coexist; worker startup and warmup are untimed.
        startup_order = rng.sample(variants, len(variants))
        for variant in startup_order:
            dataset = BufferWork(encoded, labels, variant, backend, representation)
            batches = None if workers is None else DataLoader(
                dataset, sampler=order, batch_size=32, num_workers=workers,
                persistent_workers=bool(workers), prefetch_factor=2 if workers else None,
                multiprocessing_context="spawn" if workers else None,
                worker_init_fn=worker_init, pin_memory=False, drop_last=False,
                timeout=60 if workers else 0)
            work[variant] = dataset, batches
            start = time.perf_counter_ns()
            digest = epoch(dataset, batches, order, validate=True)
            startup_ns = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            epoch(dataset, batches, order)
            warm_ns = time.perf_counter_ns() - start
            metadata[variant] = {"startup_and_validation_ns": startup_ns, "warm_epoch_ns": warm_ns,
                                 "output_sha256": digest,
                                 "worker_pids": [w.pid for w in batches._iterator._workers] if workers else []}
        if metadata["function"]["output_sha256"] != metadata["loader"]["output_sha256"]:
            raise RuntimeError("function and Loader pixels/labels differ")
        # Equal whole-epoch counts, calibrated to the faster variant with headroom.
        repetitions = max(2, math.ceil(args.seconds * 1.3e9 / min(
            item["warm_epoch_ns"] for item in metadata.values())))
        rows = []
        for round_number in range(args.rounds):
            execution_order = rng.sample(variants, len(variants))
            for position, variant in enumerate(execution_order):
                dataset, batches = work[variant]
                start = time.perf_counter_ns()
                for _ in range(repetitions):
                    epoch(dataset, batches, order)
                elapsed = time.perf_counter_ns() - start
                rows.append({"round": round_number, "position": position, "variant": variant,
                             "epochs": repetitions, "samples": len(order) * repetitions,
                             "elapsed_ns": elapsed})
                if workers:
                    pids = [w.pid for w in batches._iterator._workers]
                    if pids != metadata[variant]["worker_pids"]:
                        raise RuntimeError("persistent workers changed")
        paired = []
        for number in range(args.rounds):
            times = {row["variant"]: row["elapsed_ns"] for row in rows if row["round"] == number}
            paired.append(times["function"] / times["loader"])
        return {"backend": backend, "representation": representation,
                "workload": "direct" if workers is None else "pipeline", "workers": workers,
                "metadata": metadata, "rounds": rows, "paired_speedups": paired,
                "median_speedup": statistics.median(paired),
                "speedup_q1_q3": np.percentile(paired, [25, 75]).tolist(),
                "wins": sum(value > 1 for value in paired),
                "minimum_round_seconds": min(row["elapsed_ns"] for row in rows) / 1e9,
                "median_images_per_second": {variant: statistics.median(
                    row["samples"] * 1e9 / row["elapsed_ns"] for row in rows if row["variant"] == variant)
                    for variant in variants}}
    finally:
        for dataset, batches in work.values():
            close(batches)
        work.clear()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--cpus", default="0,1,2,3")
    args = parser.parse_args()
    if args.rounds < 3 or args.seconds <= 0:
        parser.error("use at least three rounds and positive seconds")
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    cpus = {int(value) for value in args.cpus.split(",")}
    os.sched_setaffinity(0, cpus)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hasattr(imgread.Loader, "_debug_state"):
        raise RuntimeError("timing requires a normal release build without loader-diagnostics")
    from imgread import _native
    if _native._debug_build:
        raise RuntimeError("timing requires a release build")
    if "turbojpeg" not in imgread.supported_backends():
        raise RuntimeError("auto comparison requires TurboJPEG")
    config = json.loads((Path(__file__).with_name("study.json")).read_text())["corpus"]
    config["root"] = str(args.corpus_root.resolve())
    start = time.perf_counter_ns()
    samples, _manifest = select(config)
    selection_ns = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    encoded = tuple(Path(path).read_bytes() for path, _label in samples)
    preload_ns = time.perf_counter_ns() - start
    labels = tuple(label for _path, label in samples)
    order = list(range(len(samples)))
    random.Random(37).shuffle(order)
    result = {"status": "running", "kind": "exploratory buffer comparison",
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "python": sys.version, "platform": platform.platform(),
              "cpu_model": next(line.split(":", 1)[1].strip() for line in
                                Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
              "numpy": np.__version__, "torch": torch.__version__,
              "imgread": imgread.__version__,
              "cpus": sorted(cpus), "round_count": args.rounds, "target_seconds": args.seconds,
              "corpus": config, "selection_ns": selection_ns, "preload_ns": preload_ns,
              "buffer_bytes": sum(map(len, encoded)),
              "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
              "pipeline": {"crop": [224, 224], "dtype": "uint8", "batch_size": 32,
                           "pin_memory": False, "prefetch_factor": 2, "start_method": "spawn"},
              "cells": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
    save()
    rng = random.Random(20260909)
    cells = [(backend, representation, workers) for backend in ("auto", "image")
             for representation in ("bytes", "memoryview") for workers in (None, 0, 2)]
    rng.shuffle(cells)
    try:
        for backend, representation, workers in cells:
            cell = cohort(encoded, labels, order, backend, representation, workers, args, rng)
            result["cells"].append(cell)
            save()
            print(f"{backend:5s} {representation:10s} workers={str(workers):4s} "
                  f"speedup={cell['median_speedup']:.3f} wins={cell['wins']}/{args.rounds}", flush=True)
        # Across representations and worker counts, each backend must agree too.
        for backend in ("auto", "image"):
            for workload in ("direct", "pipeline"):
                digests = {cell["metadata"]["function"]["output_sha256"] for cell in result["cells"]
                           if cell["backend"] == backend and cell["workload"] == workload}
                if len(digests) != 1:
                    raise RuntimeError("outputs differ across buffer types or worker profiles")
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
