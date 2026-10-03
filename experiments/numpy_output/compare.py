"""Matched RAM JPEG decode timing: original Vec, NumPy-owned output, simplejpeg.

Use release wheels built from the same base with the same compiler and flags.
Every route receives the same immutable JPEG bytes. The timer includes the
public call, returned NumPy array and disposal; I/O, validation and warmup are
outside it. This bounded comparison is exploratory, not corpus qualification.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import random
import statistics
import sys
import time
import warnings
from functools import partial
from pathlib import Path


def load_native(directory, alias):
    (path,) = (directory / "imgread").glob("_native*.so")
    spec = importlib.util.spec_from_file_location(f"{alias}._native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module._debug_build, path
    return module


def measure(fn, buffers, order, repeat):
    enabled = gc.isenabled()
    gc.disable()
    try:
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        for _ in range(repeat):
            for index in order:
                result = fn(buffers[index])
                del result
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
    finally:
        if enabled:
            gc.enable()
    count = len(order) * repeat
    return {"wall_us": elapsed / count / 1000, "cpu_us": cpu_elapsed / count / 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--buffers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--candidate-first",
        action="store_true",
        help="load and validate the candidate first in a fresh confirmation process",
    )
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    import numpy as np
    import simplejpeg

    sources = [("baseline", args.baseline_dir), ("candidate", args.candidate_dir)]
    if args.candidate_first:
        sources.reverse()
    natives = {label: load_native(directory, label) for label, directory in sources}
    reference = partial(
        simplejpeg.decode_jpeg, colorspace="RGB", fastdct=False, fastupsample=False
    )
    routes = {
        "functional": {
            label: partial(
                module.load_numpy_from_bytes,
                color="rgb",
                dtype="uint8",
                backend="turbojpeg",
                limits="safe",
            )
            for label, module in natives.items()
        },
        "simple": {
            label: module.load_numpy_simple_from_bytes
            for label, module in natives.items()
        },
        "loader": {
            label: module.Loader(backend="turbojpeg").decode
            for label, module in natives.items()
        },
    }
    result = {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "simplejpeg": simplejpeg.__version__,
        "modules": {label: module.__file__ for label, module in natives.items()},
        "cpu": args.cpu,
        "module_load_order": list(natives),
        "method": "balanced forward/reverse triplets",
        "endpoint": "immutable RAM JPEG bytes -> fresh NumPy RGB uint8 -> disposal",
        "timed_passes_per_sample": args.repeat,
        "block_count": args.blocks,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("error")
    for directory in sorted(args.buffers.iterdir()):
        if not directory.is_dir():
            continue
        buffers = [path.read_bytes() for path in sorted(directory.glob("*.jpg"))]
        if not buffers:
            continue
        shape = None
        for data in buffers:
            expected = reference(data)
            shape = expected.shape
            for functions in routes.values():
                for label, fn in functions.items():
                    actual = fn(data)
                    assert actual.dtype == np.uint8
                    assert actual.flags.c_contiguous and actual.flags.writeable
                    assert np.array_equal(actual, expected), (directory.name, label)
                    if label == "candidate":
                        assert actual.base is None and actual.flags.owndata
                    else:
                        assert actual.base is not None and not actual.flags.owndata
        gc.collect()
        for route, functions in routes.items():
            functions = {**functions, "simplejpeg": reference}
            blocks = []
            for block in range(args.blocks):
                rng = random.Random(88120 + block)
                image_order = rng.sample(range(len(buffers)), len(buffers))
                labels = rng.sample(list(functions), len(functions))
                call_order = labels + labels[::-1]
                samples = {label: [] for label in functions}
                for label in call_order:
                    samples[label].append(
                        measure(functions[label], buffers, image_order, args.repeat)
                    )
                means = {
                    label: statistics.mean(value["wall_us"] for value in values)
                    for label, values in samples.items()
                }
                blocks.append(
                    {
                        "order": call_order,
                        "means_us": means,
                        "saved_us": means["baseline"] - means["candidate"],
                        "saved_percent": 100
                        * (1 - means["candidate"] / means["baseline"]),
                        "candidate_over_simplejpeg_percent": 100
                        * (means["candidate"] / means["simplejpeg"] - 1),
                        "samples": samples,
                    }
                )
            row = {
                "encoder": directory.name,
                "route": route,
                "images": len(buffers),
                "shape": shape,
                "mean_input_bytes": statistics.mean(map(len, buffers)),
                "pixel_equality": "exact vs simplejpeg, all images and all routes",
                "median_us": {
                    label: statistics.median(b["means_us"][label] for b in blocks)
                    for label in functions
                },
                "median_saved_us": statistics.median(b["saved_us"] for b in blocks),
                "median_saved_percent": statistics.median(
                    b["saved_percent"] for b in blocks
                ),
                "saved_percent_range": [
                    min(b["saved_percent"] for b in blocks),
                    max(b["saved_percent"] for b in blocks),
                ],
                "positive_blocks": sum(b["saved_us"] > 0 for b in blocks),
                "candidate_over_simplejpeg_percent": statistics.median(
                    b["candidate_over_simplejpeg_percent"] for b in blocks
                ),
                "blocks": blocks,
            }
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                directory.name,
                route,
                f"saved {row['median_saved_percent']:.2f}%, "
                f"{row['positive_blocks']}/{args.blocks} blocks; "
                f"candidate vs simplejpeg {row['candidate_over_simplejpeg_percent']:+.2f}% time",
                flush=True,
            )


if __name__ == "__main__":
    main()
