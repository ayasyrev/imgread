"""Independent heaptrack prefixes; stop tracing before normal interpreter cleanup."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path

import corpus
from run import ROOT, atomic_json, checked_image, code_identity, installed_wheel, load_config, run_paths, service_envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    config = load_config(ROOT / "benchmarks/loader/study.json")
    if args.checkpoint not in config["native"]["checkpoints"]:
        parser.error("unknown checkpoint")
    os.sched_setaffinity(0, config["environment"]["cpus"])
    identity = code_identity(config)
    run_paths(config, args.run_dir, identity["code_sha"])
    service_envelope(config, identity["code_sha"], args.run_dir.name)
    provenance = json.loads((args.run_dir / "provenance.json").read_text())
    if any(identity[key] != provenance[key] for key in identity):
        raise ValueError("native probe provenance mismatch")
    installed_wheel(provenance["wheels"]["diagnostic"], True, provenance["code_sha"])
    inputs = json.loads((args.run_dir / "generated-inputs.json").read_text())["inputs"]
    for item in inputs.values():
        if corpus.digest(item["path"]) != item["sha256"]:
            raise ValueError("native input changed")
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
