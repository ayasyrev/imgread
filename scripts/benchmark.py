"""Run with uv run python scripts/benchmark.py --output result.json.

Use --legacy with the old installed implementation. Measurements use deterministic
fixtures, warmup and repeated batches; compare medians AND batch spread. This is
an opt-in release gate, not a timing assertion in shared-runner CI.
"""
import argparse
import io
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
import warnings

import numpy as np
from PIL import Image
import imgread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--max-width", type=int)
    parser.add_argument("--sources", nargs="+", choices=["path", "bytes"], default=["path", "bytes"])
    parser.add_argument("--pin-cpu", action="store_true", help="pin to one available CPU on Linux")
    args = parser.parse_args()
    if args.pin_cpu and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    rows = []
    with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for width, height in [(2, 2), (224, 224), (1920, 1080), (3840, 2160)]:
            if args.max_width is not None and width > args.max_width:
                continue
            # Textured inputs keep codec work representative and reproducible.
            data = np.random.default_rng(37).integers(0, 256, (height, width, 3), dtype="u1")
            for fmt in ("JPEG", "PNG"):
                out = io.BytesIO()
                Image.fromarray(data).save(out, format=fmt)
                encoded = out.getvalue()
                path = Path(directory) / ("image." + fmt.lower())
                path.write_bytes(encoded)
                for source in args.sources:
                    function = imgread.load_numpy if source == "path" else imgread.load_numpy_from_bytes
                    value = str(path) if source == "path" else encoded
                    profiles = ["legacy"] if args.legacy else ["safe", "unlimited"]
                    timings = {profile: [] for profile in profiles}
                    for profile in profiles:
                        kwargs = {} if args.legacy else {"limits": profile}
                        for _ in range(3): function(value, **kwargs)
                    for repeat in range(args.repeats):
                        # Alternate order to reduce temperature/frequency drift bias.
                        for profile in profiles[::1 if repeat % 2 else -1]:
                            kwargs = {} if args.legacy else {"limits": profile}
                            start = time.perf_counter_ns()
                            for _ in range(args.batch): function(value, **kwargs)
                            timings[profile].append((time.perf_counter_ns() - start) / args.batch)
                    print(f"{width}x{height} {fmt} {source} complete", flush=True)
                    for profile, samples in timings.items():
                        rows.append(dict(width=width, height=height, format=fmt, source=source,
                                         profile=profile, median_ns=statistics.median(samples),
                                         min_ns=min(samples), max_ns=max(samples), samples_ns=samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"version": imgread.__version__, "python": platform.python_version(),
                                      "platform": platform.platform(), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
