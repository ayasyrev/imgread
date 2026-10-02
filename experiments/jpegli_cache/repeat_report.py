"""Repeat an imgread_benchmark JPEG report with current and baseline Loaders.

Put the original benchmark worktree and the selected imgread wheel on PYTHONPATH.
Reuse its encoders, validation, timing loop and five-pass scheduling unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
import warnings
from dataclasses import asdict
from pathlib import Path

from experiments.ram_image_codecs.contracts import (
    DecoderSpec,
    EncoderSpec,
    atomic_json,
    read_json,
)
from experiments.ram_image_codecs.runner import (
    environment,
    measure_pass,
    schedules,
    setup_cpu,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-native", type=Path, required=True)
    parser.add_argument("--imgread-revision", required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--smoke-count", type=int)
    args = parser.parse_args()
    old = read_json(args.reference_report)
    if len(old["environment"]["affinity"]) != 1:
        parser.error("expected a single-CPU reference run")
    setup_cpu(old["environment"]["affinity"][0])
    warnings.simplefilter("error")

    import imgread
    import imgread._native
    import numpy as np
    from experiments.ram_image_codecs.codecs import (
        encode_one,
        make_decoder,
        route_metadata,
    )
    from experiments.ram_image_codecs.report import summarize_passes
    from experiments.ram_image_codecs.validation import (
        aggregate_metrics,
        check_rgb_control,
        error_metrics,
        rgb_control,
    )

    env = environment()
    assert env == old["environment"], (env, old["environment"])
    assert "turbojpeg" in imgread.supported_backends()
    native_spec = importlib.util.spec_from_file_location(
        "imgread_baseline._native",
        args.baseline_native.resolve(),
    )
    baseline = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(baseline)
    assert "turbojpeg" in baseline.supported_backends()
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = Path(old["prepared_run"])
    pixels = np.load(prepared / "prepared.npy", allow_pickle=False)
    assert len(pixels) == old["sample_size"]
    if args.smoke_count is not None:
        if not 0 < args.smoke_count <= len(pixels):
            parser.error("smoke-count must be positive and no larger than the corpus")
        pixels = pixels[: args.smoke_count]
    specifications = {row["encoder"]["id"]: row["encoder"] for row in old["rows"]}
    specs = [EncoderSpec(**item) for item in specifications.values()]
    decoders = {
        name: make_decoder(DecoderSpec(name, "jpeg"))
        for name in ("ajpegli", "simplejpeg", "imgread")
    }
    decoders["imgread_loader"] = imgread.Loader(backend="turbojpeg").decode
    decoders["imgread_loader_before"] = baseline.Loader(backend="turbojpeg").decode
    old_rows = {row["cell_id"]: row for row in old["rows"]}
    stores, rows = {}, {}
    historical_matches = 0
    encoder_matches = 0
    for spec in specs:
        print(f"encode and check: {spec.id}", flush=True)
        buffers = stores[spec.id] = [encode_one(spec, p) for p in pixels]
        if spec.family == "jpegli":
            for data, p in zip(buffers, pixels, strict=True):
                encoded = imgread.encode_jpeg(
                    p,
                    quality=spec.options["quality"],
                    subsampling=spec.options["subsampling"],
                    progressive=spec.options["progressive"],
                )
                assert encoded == data, "new encoder changed the comparison bytes"
                encoder_matches += 1
        control = rgb_control()
        control_bytes = encode_one(spec, control)
        reference = [decoders["simplejpeg"](data) for data in buffers]
        for name, decode in decoders.items():
            check_rgb_control(decode(control_bytes), control)
            metrics, differences = [], []
            held = decode(buffers[0])
            held_snapshot = held.copy()
            for data, expected, ref in zip(buffers, pixels, reference, strict=True):
                actual = decode(data)
                metrics.append(error_metrics(actual, expected))
                delta = np.abs(actual.astype(np.int16) - ref.astype(np.int16))
                differences.append((float(delta.mean()), int(delta.max())))
                if name.startswith("imgread"):
                    np.testing.assert_array_equal(actual, ref)
                    assert actual.flags.writeable
                del actual
            np.testing.assert_array_equal(held, held_snapshot)
            cid = f"{spec.id}--{name}"
            route = route_metadata(
                "imgread" if name.startswith("imgread") else name, "jpeg"
            )
            if name.startswith("imgread"):
                before = name == "imgread_loader_before"
                route.update(
                    revision=args.baseline_revision
                    if before
                    else args.imgread_revision,
                    api="Loader.decode"
                    if "loader" in name
                    else "load_numpy_from_bytes",
                    native_path=str(args.baseline_native.resolve())
                    if before
                    else imgread._native.__file__,
                )
                if "loader" in name:
                    route["copies"] = (
                        "immutable bytes borrowed; independent writable RGB NumPy output"
                    )
                    route["options"] = {
                        **route["options"],
                        "fallback_warnings": "reject via filter set outside timing",
                    }
            row = rows[cid] = {
                "cell_id": cid,
                "encoder": asdict(spec),
                "decoder": name,
                "route": route,
                "bytes_per_image": sum(map(len, buffers)) / len(buffers),
                "metrics": aggregate_metrics(metrics),
                "difference_from_simplejpeg": {
                    "mae": sum(d[0] for d in differences) / len(differences),
                    "max_abs": max(d[1] for d in differences),
                },
                "passes": [],
            }
            if args.smoke_count is None and cid in old_rows:
                previous = old_rows[cid]
                assert row["bytes_per_image"] == previous["bytes_per_image"]
                assert row["metrics"] == previous["metrics"]
                assert (
                    row["difference_from_simplejpeg"]
                    == previous["difference_from_simplejpeg"]
                )
                historical_matches += 1
        del reference

    orders = schedules(list(rows), len(pixels))
    for row in rows.values():
        for data in stores[row["encoder"]["id"]]:
            output = decoders[row["decoder"]](data)
            del output
    for order in orders:
        print(f"timed pass {order['pass_id'] + 1}/5", flush=True)
        for cid in order["cells"]:
            row = rows[cid]
            result = measure_pass(
                decoders[row["decoder"]], stores[row["encoder"]["id"]], order["images"]
            )
            assert result.status == "successful" and result.processed == len(pixels), (
                result
            )
            row["passes"].append({"pass_id": order["pass_id"], **asdict(result)})
    for row in rows.values():
        row["speed"] = summarize_passes(row["passes"], len(pixels))

    # Confirm the small gaps with matched image orders and reversed reader order.
    paired = []
    rng = random.Random(20261002)
    names = ["simplejpeg", "imgread_loader", "imgread_loader_before"]
    for spec in specs:
        print(f"paired confirmation: {spec.id}", flush=True)
        records = []
        for round_id in range(5):
            order = rng.sample(range(len(pixels)), len(pixels))
            methods = rng.sample(names, len(names))
            for direction, method_order in (("AB", methods), ("BA", methods[::-1])):
                for name in method_order:
                    timing = measure_pass(decoders[name], stores[spec.id], order)
                    assert timing.status == "successful" and timing.processed == len(
                        pixels
                    ), timing
                    records.append(
                        {
                            "round": round_id,
                            "direction": direction,
                            "decoder": name,
                            **asdict(timing),
                        }
                    )
        ratios = {name: [] for name in names if name != "imgread_loader"}
        for round_id in range(5):
            totals = {
                name: sum(
                    r["elapsed_ns"]
                    for r in records
                    if r["round"] == round_id and r["decoder"] == name
                )
                for name in names
            }
            for name, values in ratios.items():
                values.append(totals[name] / totals["imgread_loader"])
        paired.append(
            {
                "encoder_id": spec.id,
                "records": records,
                "current_loader_speedup": ratios,
            }
        )

    result = {
        "complete": True,
        "reference_report": str(args.reference_report.resolve()),
        "prepared_run": str(prepared),
        "sample_size": len(pixels),
        "shape": list(pixels.shape[1:]),
        "environment": env,
        "imgread_revision": args.imgread_revision,
        "baseline_revision": args.baseline_revision,
        "historical_quality_and_size_matches": historical_matches,
        "new_encoder_byte_matches": encoder_matches,
        "orders": orders,
        "rows": list(rows.values()),
        "paired": paired,
        "note": old["note"]
        + " Original imgread adapter retained, including per-call warning context. Added direct current/baseline Loader.decode and five AB/BA confirmation rounds. JPEG buffers encoded afresh from the original prepared corpus; every reader shares the same byte objects in this run.",
    }
    atomic_json(args.output / "report.json", result)
    lines = [
        "# JPEG decoder comparison with updated imgread",
        "",
        f"N={len(pixels)}, RGB uint8 256x256, CPU affinity={env['affinity']}.",
        f"imgread {args.imgread_revision}; baseline Loader {args.baseline_revision}.",
        result["note"],
        "",
        f"Original size/quality rows reproduced exactly: {historical_matches}. New encode_jpeg byte matches: {encoder_matches}.",
        "",
        "## Throughput summary",
        "",
        "Median images/s from the original five-pass protocol. The imgread function retains its original adapter; Loader calls are direct.",
        "",
        "| Encoder | KiB/image | PSNR via imgread | ajpegli | simplejpeg | imgread function | New Loader | Previous Loader |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in specs:
        group = {name: rows[f"{spec.id}--{name}"] for name in decoders}
        current = group["imgread_loader"]
        rates = " | ".join(
            f"{group[name]['speed']['median_images_s']:.1f}" for name in decoders
        )
        lines.append(
            f"| {spec.id} | {current['bytes_per_image'] / 1024:.2f} | {current['metrics']['psnr']['value']:.2f} | {rates} |"
        )
    lines += [
        "",
        "## Paired confirmation",
        "",
        "Median and range of five AB/BA paired speed ratios. Positive percentages favor the new Loader. If the range crosses zero, a difference is not established.",
        "",
        "| Encoder | New Loader vs simplejpeg, % [min, max] | New vs previous Loader, % [min, max] |",
        "|---|---:|---:|",
    ]
    for pair in paired:
        cells = []
        for name in ("simplejpeg", "imgread_loader_before"):
            values = [(v - 1) * 100 for v in pair["current_loader_speedup"][name]]
            cells.append(
                f"{statistics.median(values):+.2f} [{min(values):+.2f}, {max(values):+.2f}]"
            )
        lines.append(f"| {pair['encoder_id']} | {' | '.join(cells)} |")
    lines += [
        "",
        "## Detailed original-format rows",
        "",
        "| Encoder / decoder | images/s median [min, max] | KiB/image | PSNR | MAE vs source | MAE / max vs simplejpeg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows.values():
        speed, quality = row["speed"], row["metrics"]
        diff = row["difference_from_simplejpeg"]
        lines.append(
            f"| {row['cell_id']} | {speed['median_images_s']:.1f} [{speed['min_images_s']:.1f}, {speed['max_images_s']:.1f}] | {row['bytes_per_image'] / 1024:.2f} | {quality['psnr']['value']:.2f} | {quality['mae']:.3f} | {diff['mae']:.3f} / {diff['max_abs']} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(args.output / "report.md", flush=True)


if __name__ == "__main__":
    main()
