"""RAM JPEG cache comparison: identical bytes, RGB output, balanced AB/BA timing.

Use a prepared N,H,W,3 uint8 .npy corpus. Preparation, encoding, correctness
checks and warmup are outside timing. Each timed call includes allocation and
output disposal. Baseline is an optional, separately built imgread extension.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time
import warnings

import ajpegli
import numpy as np
import simplejpeg

import imgread


def measure(decode, buffers, order):
    start = time.perf_counter_ns()
    for index in order:
        pixels = decode(buffers[index])
        del pixels
    return time.perf_counter_ns() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-native", type=Path)
    parser.add_argument("--baseline-label", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    if args.count < 1 or args.rounds < 1:
        parser.error("count and rounds must be positive")
    os.sched_setaffinity(0, {args.cpu})
    warnings.simplefilter("error")
    source = np.load(args.prepared, mmap_mode="r", allow_pickle=False)
    if source.dtype != np.uint8 or source.ndim != 4 or source.shape[-1] != 3:
        parser.error("prepared corpus must be N,H,W,3 uint8")
    if len(source) < args.count:
        parser.error("prepared corpus contains fewer images than count")
    # Materialize once, so neither mmap faults nor input I/O enter timing.
    source = np.array(source[:args.count], copy=True)
    if "turbojpeg" not in imgread.supported_backends():
        raise RuntimeError("current imgread must be built with TurboJPEG")
    decoders = {
        "imgread_function": imgread.load_numpy_from_bytes,
        "imgread_loader": imgread.Loader().decode,
        # simplejpeg defaults: RGB, accurate DCT, fancy upsampling, strict input.
        "simplejpeg": simplejpeg.decode_jpeg,
    }
    if args.baseline_native:
        spec = importlib.util.spec_from_file_location(
            "imgread_baseline._native", args.baseline_native.resolve(),
        )
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        if "turbojpeg" not in baseline.supported_backends():
            raise RuntimeError("baseline imgread must be built with TurboJPEG")
        decoders["baseline_function"] = baseline.load_numpy_from_bytes
        decoders["baseline_loader"] = baseline.Loader().decode

    configs = [
        {"encoder": "jpegli", "quality": q, "subsampling": "420", "progressive": 0}
        for q in (50, 75, 90, 95)
    ] + [
        {"encoder": "jpegli", "quality": 90, "subsampling": "420", "progressive": 2},
        {"encoder": "jpegli", "quality": 90, "subsampling": "444", "progressive": 0},
        {"encoder": "simplejpeg", "quality": 90, "subsampling": "420", "progressive": 0},
    ]
    report = {
        "complete": False,
        "prepared": str(args.prepared.resolve()),
        "shape": list(source.shape),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "imgread_version": imgread.__version__,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "ajpegli", "simplejpeg")},
        "jpegli_commit": ajpegli.__jpegli_commit__,
        "baseline_label": args.baseline_label,
        "baseline_native": str(args.baseline_native) if args.baseline_native else None,
        "rounds": args.rounds,
        "contract": "Immutable encoded bytes in RAM -> new writable contiguous RGB uint8 NumPy array; single CPU. Encode, validation, warmup and I/O excluded. RAM lookup, public call and output disposal included. Five AB/BA rounds by default. Decoder fast modes disabled. Warnings treated as errors outside the timed loop.",
        "rows": [],
    }
    rng = random.Random(20261002)
    for config in configs:
        options = {k: v for k, v in config.items() if k != "encoder"}
        print(f"Encode and validate {config}", flush=True)
        started = time.perf_counter()
        if config["encoder"] == "jpegli":
            buffers = [imgread.encode_jpeg(pixels, **options) for pixels in source]
        else:
            buffers = [simplejpeg.encode_jpeg(
                pixels, quality=config["quality"], colorsubsampling=config["subsampling"],
            ) for pixels in source]
        encode_seconds = time.perf_counter() - started
        total_absolute = total_squared = 0.0
        max_difference = {name: 0 for name in decoders}
        for original, data in zip(source, buffers, strict=True):
            expected = decoders["imgread_loader"](data)
            error = expected.astype(np.float64) - original
            total_absolute += np.abs(error).sum()
            total_squared += np.square(error).sum()
            for name, decode in decoders.items():
                actual = decode(data)
                assert actual.shape == original.shape and actual.dtype == np.uint8
                assert actual.flags.c_contiguous and actual.flags.writeable
                difference = int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max())
                max_difference[name] = max(max_difference[name], difference)
                if name != "simplejpeg":
                    np.testing.assert_array_equal(actual, expected)
        # Check output lifetime across calls and mutation separately from timing.
        for decode in decoders.values():
            first = decode(buffers[0])
            saved = first.copy()
            second = decode(buffers[-1])
            second.fill(0)
            np.testing.assert_array_equal(first, saved)
            for data in buffers:
                pixels = decode(data)
                del pixels
        row = {
            **config, "bytes_per_image": sum(map(len, buffers)) / len(buffers),
            "encode_seconds": encode_seconds,
            "mae": total_absolute / source.size,
            "psnr": 10 * math.log10(255**2 / (total_squared / source.size)) if total_squared else None,
            "max_difference_from_current_loader": max_difference,
            "timing": [],
        }
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for round_id in range(args.rounds):
                order = list(range(len(buffers)))
                rng.shuffle(order)
                methods = list(decoders)
                rng.shuffle(methods)
                # Match each image order and reverse the decoder order to
                # balance CPU drift; preserve both observations in the report.
                for direction, names in (("AB", methods), ("BA", methods[::-1])):
                    for name in names:
                        elapsed = measure(decoders[name], buffers, order)
                        row["timing"].append({
                            "round": round_id, "direction": direction,
                            "decoder": name, "elapsed_ns": elapsed,
                            "images_per_second": len(buffers) * 1e9 / elapsed,
                        })
        finally:
            if was_enabled:
                gc.enable()
        row["throughput"] = {}
        for name in decoders:
            values = [t["images_per_second"] for t in row["timing"] if t["decoder"] == name]
            row["throughput"][name] = {
                "median": statistics.median(values), "min": min(values), "max": max(values),
            }
        row["paired_speedup"] = {}
        for reference in ("baseline_loader", "simplejpeg"):
            if reference not in decoders:
                continue
            ratios = []
            for round_id in range(args.rounds):
                times = {
                    name: sum(t["elapsed_ns"] for t in row["timing"]
                              if t["round"] == round_id and t["decoder"] == name)
                    for name in (reference, "imgread_loader")
                }
                ratios.append(times[reference] / times["imgread_loader"])
            row["paired_speedup"][reference] = ratios
        report["rows"].append(row)
        print({name: round(value["median"]) for name, value in row["throughput"].items()}, flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
