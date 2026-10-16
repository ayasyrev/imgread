"""Paired public-API timings on identical RAM JPEGs, with fresh owned outputs."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import random
import resource
import statistics
import sys
import time
import warnings
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(function, buffers, order, repeat):
    enabled = gc.isenabled()
    gc.disable()
    try:
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        cpu = time.process_time_ns()
        start = time.perf_counter_ns()
        for _ in range(repeat):
            for index in order:
                result = function(buffers[index])
                del result
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults
    finally:
        if enabled:
            gc.enable()
    count = len(order) * repeat
    return {"wall_us": elapsed / count / 1000, "cpu_us": cpu_elapsed / count / 1000,
            "minor_faults": faults}


def validate(functions, reference, buffers, offset):
    import numpy as np

    shapes = set()
    for data in buffers:
        expected = reference(data)
        shapes.add(expected.shape)
        for name, function in functions.items():
            actual = function(data)
            assert np.array_equal(actual, expected), name
            assert actual.dtype == np.uint8 and actual.flags.c_contiguous
            assert actual.flags.writeable and actual.flags.owndata and actual.base is None
            if offset is not None:
                assert actual.ctypes.data % 64 == offset
    for name, function in functions.items():
        first = function(buffers[0])
        saved = first.copy()
        second = function(buffers[-1])
        second.fill(0)
        assert np.array_equal(first, saved), name
    return {"images": len(buffers), "shapes": sorted(shapes),
            "exact_rgb": True, "fresh_owned_writable_output": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--buffers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simplejpeg-extension", type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--alignment-module", type=Path)
    parser.add_argument("--offset", type=int, choices=(0, 16))
    parser.add_argument("--cpu", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--count", type=int)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--candidate-first", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if bool(args.alignment_module) != (args.offset is not None):
        parser.error("alignment-module and offset must be supplied together")
    os.sched_setaffinity(0, {args.cpu})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import simplejpeg

    labels = ["baseline", "candidate"]
    if args.candidate_first:
        labels.reverse()
    natives = {
        name: load(next((args.build / name / "imgread").glob("_native*.so")), f"{name}._native")
        for name in labels
    }
    assert all(not module._debug_build for module in natives.values())
    comparator = (
        load(args.simplejpeg_extension, "common._jpeg")
        if args.simplejpeg_extension else simplejpeg
    )
    functions = {}
    for name, module in natives.items():
        functions[f"{name}.function"] = module.load_numpy_from_bytes
        functions[f"{name}.simple"] = module.load_numpy_simple_from_bytes
        functions[f"{name}.loader"] = module.Loader().decode
    functions["simplejpeg"] = comparator.decode_jpeg
    maps = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                   if "libturbojpeg" in line and "/" in line})
    if args.native_library:
        assert maps == [str(args.native_library.resolve())], maps
    record = {
        "command": sys.argv, "python": sys.version, "numpy": np.__version__,
        "simplejpeg": simplejpeg.__version__, "platform": platform.platform(),
        "cpu": args.cpu, "build": json.loads((args.build / "build.json").read_text()),
        "module_paths": {name: module.__file__ for name, module in natives.items()},
        "simplejpeg_path": comparator.__file__, "native_runtime_maps": maps,
        "allocation": "default" if args.offset is None else f"private mod64={args.offset}",
        "endpoint": "RAM lookup + public decode + fresh owned RGB output + disposal",
        "protocol": "paired shuffled forward/reverse blocks; validation returns before timing",
        "blocks": args.blocks, "passes_per_sample": args.repeat,
        "rows": [],
    }
    policy = load(args.alignment_module, "_imgread_alignment_policy") if args.alignment_module else None
    previous = policy.install(args.offset) if policy else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("error")
    try:
        directories = sorted(path for path in args.buffers.iterdir() if path.is_dir())
        for directory in directories:
            if args.cases and directory.name not in args.cases:
                continue
            paths = sorted(directory.glob("*.jpg"))
            if args.count:
                paths = paths[:args.count]
            if not paths:
                continue
            buffers = [path.read_bytes() for path in paths]
            checked = validate(functions, simplejpeg.decode_jpeg, buffers, args.offset)
            # validate's arrays are dead; warmup/disposal has the same pattern for all routes.
            order = list(range(len(buffers)))
            for function in functions.values():
                measure(function, buffers, order, 1)
            gc.collect()
            blocks = []
            for block in range(args.blocks):
                rng = random.Random(44123 + block + 500 * args.candidate_first)
                order = rng.sample(range(len(buffers)), len(buffers))
                names = rng.sample(list(functions), len(functions))
                samples = {name: [] for name in names}
                for name in names + names[::-1]:
                    samples[name].append(measure(functions[name], buffers, order, args.repeat))
                means = {name: statistics.mean(s["wall_us"] for s in values)
                         for name, values in samples.items()}
                cpu_means = {name: statistics.mean(s["cpu_us"] for s in values)
                             for name, values in samples.items()}
                blocks.append({"order": names + names[::-1], "samples": samples,
                               "wall_us": means, "cpu_us": cpu_means})
            pairs = {}
            for route in ("function", "simple", "loader"):
                for other in (f"baseline.{route}", "simplejpeg"):
                    candidate = f"candidate.{route}"
                    ratios = [100 * (b["wall_us"][candidate] / b["wall_us"][other] - 1)
                              for b in blocks]
                    pairs[f"{candidate}_over_{other}"] = {
                        "median_percent": statistics.median(ratios),
                        "range_percent": [min(ratios), max(ratios)],
                        "candidate_slower_blocks": sum(value > 0 for value in ratios),
                    }
            row = {"case": directory.name, "inputs": [str(path) for path in paths],
                   "validation": checked, "mean_encoded_bytes": statistics.mean(map(len, buffers)),
                   "median_us": {name: statistics.median(b["wall_us"][name] for b in blocks)
                                 for name in functions}, "pairs": pairs, "blocks": blocks}
            record["rows"].append(row)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(args.output.stem, directory.name, {key: round(value["median_percent"], 3)
                                  for key, value in pairs.items()}, flush=True)
    finally:
        if policy:
            policy.restore(previous)
    if not record["rows"]:
        raise RuntimeError("no JPEG inputs selected")
    record["complete"] = True
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
