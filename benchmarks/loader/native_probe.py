"""Independent heaptrack prefixes; stop tracing before normal interpreter cleanup."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path

from run import atomic_json, checked_image, installed_wheel, load_config, service_envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    config = load_config(args.run_dir / "study.json")
    if args.checkpoint not in config["native"]["checkpoints"]:
        parser.error("unknown checkpoint")
    os.sched_setaffinity(0, config["environment"]["cpus"])
    service_envelope(config)
    installed_wheel(True)
    inputs = json.loads((args.run_dir / "generated-inputs.json").read_text())["inputs"]
    import imgread
    loader = imgread.Loader(backend="auto", max_buffer_bytes=1048576)
    records = []
    for phase, kind, count in config["sequence"]["phases"]:
        for _ in range(count):
            image = checked_image(loader, inputs[kind]["path"], kind)
            del image
        records.append({"phase": phase, "diagnostics": loader._debug_state()})
        if phase == args.checkpoint:
            break
    if args.checkpoint == "destroyed":
        del loader
        gc.collect()
    path = args.run_dir / "native" / f"{args.checkpoint}-{args.repeat:02d}.diagnostics.json"
    atomic_json(path, {"checkpoint": args.checkpoint, "repeat": args.repeat, "records": records, "destroyed": args.checkpoint == "destroyed"})
    # heaptrack preloads this exported void(void) function in this child only.
    stop = ctypes.CDLL(None).heaptrack_stop
    stop.argtypes = []
    stop.restype = None
    stop()
    # Loader deliberately stays alive until tracing stops at non-destroyed prefixes.


if __name__ == "__main__":
    main()
