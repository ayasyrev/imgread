"""Reject incomplete evidence, then compute paired timing and memory results."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

import corpus
from run import STAGES, atomic_json, canonical, configurations, load_config, run_paths, verify_stage


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def distribution(values):
    if not values:
        raise ValueError("empty distribution")
    ordered = sorted(values)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else (ordered[0],) * 3
    return {"values": values, "median": statistics.median(values), "iqr": q3 - q1, "min": min(values), "max": max(values)}


def compare(candidate, baseline, policy):
    ratios = [left / right for left, right in zip(candidate, baseline, strict=True)]
    median = statistics.median(ratios)
    return {"ratios": distribution(ratios),
            "material_regression": median > 1 + policy["material_slowdown"] and sum(ratio > 1 for ratio in ratios) >= policy["consistent_rounds"],
            "reliable_acceleration": median <= 1 - policy["minimum_improvement"] and sum(ratio < 1 for ratio in ratios) >= policy["consistent_rounds"]}


def break_even(preparation_index, preparation_baseline, epoch_index, epoch_baseline):
    if epoch_baseline <= epoch_index:
        return None
    return max(0, math.ceil((preparation_index - preparation_baseline) / (epoch_baseline - epoch_index)))


def timings(config, rows, validation):
    matrix = {cell["config_id"]: cell for cell in configurations(config)}
    grouped = defaultdict(list)
    for row in rows:
        if row["config_id"] not in matrix or row["round"] not in range(5):
            raise ValueError("unexpected timing cell or round")
        if row["binding"] != validation["binding"] or row["order_digest"] != validation["order_digest"]:
            raise ValueError("timing provenance/order mismatch")
        if row["result_digest"] != validation["cells"][row["config_id"]]["result_digest"]:
            raise ValueError("timing result digest mismatch")
        for key, value in matrix[row["config_id"]].items():
            if row[key] != value:
                raise ValueError("timing configuration fields disagree")
        for name in ("elapsed_ns", "samples", "startup_ns", "preparation_ns", "epoch"):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError("timings require non-negative integer counts/nanoseconds")
        if row["elapsed_ns"] == 0 or row["samples"] != config["corpus"]["classes"] * config["corpus"]["per_class"]:
            raise ValueError("timing sample count mismatch")
        grouped[row["config_id"], row["round"]].append(row)
    if set(grouped) != {(cell, round_) for cell in matrix for round_ in range(5)}:
        raise ValueError("incomplete timing matrix")
    aggregates = {}
    for cell_id, cell in matrix.items():
        per_round = []
        for round_ in range(5):
            values = sorted(grouped[cell_id, round_], key=lambda row: row["epoch"])
            if len(values) < 2 or [row["epoch"] for row in values] != list(range(len(values))):
                raise ValueError("duplicate/missing epochs or fewer than two epochs")
            total = sum(row["elapsed_ns"] for row in values)
            if total < 2_000_000_000:
                raise ValueError("short timing round (<2 seconds)")
            sizes = {len(grouped[other, round_]) for other, other_cell in matrix.items()
                     if all(cell[key] == other_cell[key] for key in ("workload", "backend", "workers"))}
            if len(sizes) != 1:
                raise ValueError("variants used unequal epoch counts")
            per_round.append({"round": round_, "ns_per_image": total / sum(row["samples"] for row in values),
                              "epoch_ns": total / len(values), "startup_ns": values[0]["startup_ns"],
                              "preparation_ns": values[0]["preparation_ns"]})
        aggregates[cell_id] = dict(cell, rounds=per_round,
                                  ns_per_image=distribution([row["ns_per_image"] for row in per_round]),
                                  images_per_second=distribution([1e9 / row["ns_per_image"] for row in per_round]),
                                  startup_ns=distribution([row["startup_ns"] for row in per_round]),
                                  preparation_ns=distribution([row["preparation_ns"] for row in per_round]))
    effects = {}
    for cell_id, cell in matrix.items():
        if cell["variant"] == "function":
            continue
        for baseline in (["function", "path"] if cell["variant"] == "index" else ["function"]):
            other_id = cell_id.rsplit("-", 1)[0] + "-" + baseline
            candidate, reference = aggregates[cell_id], aggregates[other_id]
            effect = compare(candidate["ns_per_image"]["values"], reference["ns_per_image"]["values"], config["decision"])
            effect["break_even_epochs"] = [break_even(left["startup_ns"] + left["preparation_ns"], right["startup_ns"] + right["preparation_ns"], left["epoch_ns"], right["epoch_ns"])
                                            for left, right in zip(candidate["rounds"], reference["rounds"], strict=True)]
            effect["preparation_delta_ns"] = [left["startup_ns"] + left["preparation_ns"] - right["startup_ns"] - right["preparation_ns"]
                                                for left, right in zip(candidate["rounds"], reference["rounds"], strict=True)]
            effects[cell_id + " vs " + baseline] = effect
    return {"configurations": aggregates, "effects": effects}


def native_bytes(path):
    sites = defaultdict(int)
    all_bytes = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            stack, value = line.rsplit(" ", 1)
            count = int(value)
        except (ValueError, TypeError) as error:
            raise ValueError("malformed heaptrack folded stack") from error
        if count < 0:
            raise ValueError("negative allocation cost in single-run profile")
        all_bytes += count
        # Frames are root-to-leaf. Classify once at the nearest native allocation
        # site, so nested tj/libjpeg frames never double-count a single allocation.
        frames = stack.split(";")
        native_context = re.search(r"jpeg_|tj3|jinit_", stack)
        site = next((frame for frame in reversed(frames) if re.search(r"jpeg_(?:get|alloc)|alloc_(?:small|large|sarray|barray)|jinit_memory_mgr|tj3Init", frame)), None) if native_context else None
        if site is not None:
            sites[site] += count
    return {"native_bytes": sum(sites.values()), "all_bytes": all_bytes, "sites": dict(sites)}


def native_evidence(config, run_dir):
    manifest = json.loads((run_dir / "native-profiles.json").read_text())
    expected = {(checkpoint, repeat) for checkpoint in config["native"]["checkpoints"] for repeat in range(1, 4)}
    if len(manifest) != len(expected) or {(row["checkpoint"], row["repeat"]) for row in manifest} != expected:
        raise ValueError("incomplete native checkpoint/repeat matrix")
    results = []
    for row in manifest:
        if not Path(row["profile"]).is_file():
            raise ValueError("native profile missing")
        measured = {}
        for label, export in row["exports"].items():
            measured[label] = native_bytes(export["path"])
        if set(measured) != {"live", "peak"} or not any(value["sites"] for value in measured.values()):
            raise ValueError("missing symbolized native attribution (not zero native memory)")
        diagnostics = json.loads((run_dir / "native" / f"{row['checkpoint']}-{row['repeat']:02d}.diagnostics.json").read_text())
        if diagnostics["checkpoint"] != row["checkpoint"] or diagnostics["repeat"] != row["repeat"]:
            raise ValueError("native diagnostics binding mismatch")
        phases = [phase[0] for phase in config["sequence"]["phases"]]
        expected_phases = phases if row["checkpoint"] == "destroyed" else phases[:phases.index(row["checkpoint"]) + 1]
        if [record["phase"] for record in diagnostics["records"]] != expected_phases:
            raise ValueError("native probe did not execute the complete fixed prefix")
        for record in diagnostics["records"]:
            state = record["diagnostics"]
            if state["input_len"] or state["input_capacity"] > 1048576 or state["native_live"] > 1:
                raise ValueError("native prefix resource invariant failed")
        if row["checkpoint"] not in ("destroyed", "corrupt") and measured["live"]["native_bytes"] == 0:
            raise ValueError("missing retained native attribution for a live handle")
        if row["checkpoint"] == "destroyed" and measured["live"]["native_bytes"]:
            raise ValueError("native allocations remain after Loader destruction")
        results.append(dict(row, measured=measured))
    return results


def memory_evidence(config, rows, run_dir):
    import multiprocessing as mp
    expected_manifest = {(length, workers, method, variant, repeat)
                         for length in config["manifest"]["lengths"] for workers in (0, 2)
                         for method in ([value for value in config["manifest"]["start_methods"] if value in mp.get_all_start_methods()] if workers else [None])
                         for variant in config["variants"] for repeat in range(3)}
    manifests = defaultdict(list)
    sequences = defaultdict(list)
    for row in rows:
        for key in ("rss_bytes", "pss_bytes", "peak_rss_bytes"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("missing process memory observation")
        if row["kind"] == "manifest":
            manifests[row["length"], row["workers"], row["method"], row["variant"], row["repeat"]].append(row)
        elif row["kind"] == "sequence":
            sequences[row["backend"], row["cap"], row["workers"]].append(row)
        else:
            raise ValueError("unexpected memory scenario")
    if set(manifests) != expected_manifest:
        raise ValueError("incomplete manifest matrix")
    for key, values in manifests.items():
        if len(values) != (1 + key[1]) * 2:
            raise ValueError("incomplete manifest process phases")
        for row in values:
            diagnostics = row["diagnostics"]
            if diagnostics["input_capacity"] or diagnostics["native_live"]:
                raise ValueError("manifest-only construction allocated decoder resources")
            expected_entries = key[0] if key[3] == "index" else 0
            if diagnostics.get("manifest_entries", 0) != expected_entries:
                raise ValueError("manifest ownership differs from variant")
    expected_sequences = {(backend, cap, workers) for backend in config["backends"] for cap in (0, 1048576) for workers in (0, 2)}
    if set(sequences) != expected_sequences:
        raise ValueError("incomplete sequence memory matrix")
    for (backend, cap, workers), values in sequences.items():
        active = [row for row in values if row["role"] == ("worker" if workers else "parent")]
        pids = {row["pid"] for row in active}
        if len(pids) != max(1, workers):
            raise ValueError("sequence persistent worker PID changed")
        expected_phases = {(epoch, phase[0]) for epoch in range(2) for phase in config["sequence"]["phases"]}
        expected_phases |= {(1, phase) for phase in ("outputs-retained", "outputs-released", "destroyed")}
        for pid in pids:
            observed = [row for row in active if row["pid"] == pid]
            if len(observed) != len(expected_phases) or {(row["epoch"], row["phase"]) for row in observed} != expected_phases:
                raise ValueError("incomplete per-worker sequence")
            for row in observed:
                states = [call["diagnostics"] for call in row.get("calls", [])] or [row.get("diagnostics", {})]
                for state in states:
                    if state.get("input_len", 0) or state.get("input_capacity", 0) > cap or state.get("native_live", 0) > 1:
                        raise ValueError("retained input/native resource violation")
                output_count = config["sequence"]["retained_outputs"]
                output_nbytes = output_count * math.prod(config["sequence"]["small_shape"])
                if row["phase"] == "outputs-retained" and (row["output_count"] != output_count or row["output_nbytes"] != output_nbytes):
                    raise ValueError("retained output control mismatch")
                if len(states) == 32 and backend == "auto":
                    if {state["native_live"] for state in states} != {1} or len({state["native_creations"] for state in states}) != 1:
                        raise ValueError("missing native reuse in warmed small sequence")
    sampling = list((run_dir / "jobs").glob("*.samples.jsonl"))
    if len(sampling) < len(manifests) + len(sequences):
        raise ValueError("100ms process sampling missing")
    peaks = {}
    for path in sampling:
        samples = read_rows(path)
        if not samples:
            raise ValueError("empty process memory sampling file")
        by_pid, by_tick = defaultdict(list), defaultdict(list)
        for row in samples:
            by_pid[row["pid"]].append(row)
            by_tick[row["sample_ns"]].append(row)
        peaks[path.name] = {"per_pid": {pid: {key: max(row[key] for row in values) for key in ("rss_bytes", "pss_bytes", "peak_rss_bytes")} for pid, values in by_pid.items()},
                           "process_tree_rss_peak": max(sum(row["rss_bytes"] for row in values) for values in by_tick.values()),
                           "process_tree_pss_peak": max(sum(row["pss_bytes"] for row in values) for values in by_tick.values())}
    manifest_summary = {}
    for length, workers, method, variant, _repeat in sorted(manifests, key=str):
        name = f"n{length}-w{workers}-{method}-{variant}"
        if name in manifest_summary:
            continue
        values = [row for key, group in manifests.items() if key[:4] == (length, workers, method, variant) for row in group if row["role"] == "parent" and row["phase"] == 0]
        manifest_summary[name] = {key: distribution([row["preparation"][key] for row in values]) for key in ("list_ns", "construction_ns", "pickle_bytes", "pickle_ns", "startup_ns")}
        manifest_summary[name].update({key: distribution([row[key] for row in values]) for key in ("rss_bytes", "pss_bytes", "peak_rss_bytes")})
        manifest_summary[name]["manifest_bytes"] = distribution([row["diagnostics"]["manifest_bytes"] for row in values])
        manifest_summary[name]["worker_memory"] = [dict(pid=row["pid"], repeat=key[-1], phase=row["phase"], rss_bytes=row["rss_bytes"], pss_bytes=row["pss_bytes"], manifest_bytes=row["diagnostics"]["manifest_bytes"])
                                                    for key, group in manifests.items() if key[:4] == (length, workers, method, variant) for row in group if row["role"] == "worker"]
    sequence_summary = []
    for (backend, cap, workers), values in sequences.items():
        phases = defaultdict(list)
        for row in values:
            phases[row["epoch"], row["phase"]].append(row)
        for (epoch, phase), observations in phases.items():
            states = [call["diagnostics"] for row in observations for call in row.get("calls", [])]
            states += [row["diagnostics"] for row in observations if "diagnostics" in row]
            sequence_summary.append({"backend": backend, "cap": cap, "workers": workers, "epoch": epoch, "phase": phase,
                                     "rss_sum": sum(row["rss_bytes"] for row in observations), "pss_sum": sum(row["pss_bytes"] for row in observations),
                                     "input_capacity_max": max((state.get("input_capacity", 0) for state in states), default=0),
                                     "input_peak_capacity_max": max((state.get("input_peak_capacity", 0) for state in states), default=0),
                                     "native_live_per_loader_max": max((state.get("native_live", 0) for state in states), default=0),
                                     "output_nbytes": sum(row.get("output_nbytes", 0) for row in observations)})
    return {"manifest_cases": len(manifests), "sequence_cases": len(sequences), "sample_peaks": peaks,
            "manifest_summary": manifest_summary, "sequence_summary": sequence_summary,
            "output_nbytes_max": max(row.get("output_nbytes", 0) for row in rows)}


def memory_admission(config, provenance):
    admission = provenance["resource_preflight"]
    required = config["resources"]["min_available_bytes"]
    if (admission["required_bytes"] != required
            or admission["allow_low_memory"] is not provenance["binding"].get("allow_low_memory", False)
            or admission["threshold_met"] is not (admission["available_bytes"] >= required)
            or (not admission["threshold_met"] and not admission["allow_low_memory"])):
        raise ValueError("memory admission provenance mismatch")
    if admission["allow_low_memory"]:
        return [f"Owner-authorized low-memory run: preflight observed {admission['available_bytes'] / 2**30:.3f} GiB available RAM; the default admission threshold is {required / 2**30:.0f} GiB. The threshold was waived for this attempt. Results require confirmation on another machine with sufficient memory; the measurement matrix and other resource limits are unchanged."]
    return []


def generate(config, run_dir, check_complete=True):
    run_dir = Path(run_dir).resolve()
    provenance = json.loads((run_dir / "provenance.json").read_text())
    _, control = run_paths(run_dir)
    stages = json.loads((control / "stages.json").read_text())
    for stage in STAGES[:-1]:
        if stage not in stages:
            raise ValueError("report requires all five complete predecessor stages")
        verify_stage(stages[stage], run_dir, provenance["binding"])
    if canonical(json.loads((run_dir / "study.json").read_text())) != canonical(config):
        raise ValueError("saved config differs from report settings")
    limitations = memory_admission(config, provenance)
    validation = json.loads((run_dir / "validation.json").read_text())
    if validation["binding"] != provenance["binding"]:
        raise ValueError("validation/build binding mismatch")
    if len(provenance.get("pipeline_constructors", [])) != 12:
        raise ValueError("preflight did not construct all twelve DataLoader cells")
    config_ids = {cell["config_id"] for cell in configurations(config)}
    constructors = {row["config_id"]: row["kwargs"] for row in provenance["pipeline_constructors"]}
    from run import pipeline_kwargs, check_validation_digests
    for cell in configurations(config):
        if cell["workload"] == "pipeline" and constructors.get(cell["config_id"]) != pipeline_kwargs(config, cell["workers"]):
            raise ValueError("recorded pipeline constructor settings changed")
    stress_keys = [(row["backend"], row["kind"]) for row in validation["stress"]]
    expected_stress = {(backend, kind) for backend in config["backends"] for kind in corpus.SYNTHETIC_FILES}
    if set(validation["cells"]) != config_ids or len(stress_keys) != len(expected_stress) or set(stress_keys) != expected_stress:
        raise ValueError("incomplete output validation")
    check_validation_digests(config, validation["cells"])
    timing = timings(config, read_rows(run_dir / "timings.jsonl"), validation)
    memory = memory_evidence(config, read_rows(run_dir / "memory.jsonl"), run_dir)
    native = native_evidence(config, run_dir)
    regressions = [name for name, effect in timing["effects"].items() if effect["material_regression"]]
    summary = {"complete": True, "binding": provenance["binding"], "timing": timing, "memory": memory, "native": native,
               "resource_preflight": provenance["resource_preflight"], "execution_limitations": limitations,
               "decision": "return-to-owner-material-regression" if regressions else "completed", "regressions": regressions}
    atomic_json(run_dir / "summary.json", summary)
    lines = ["# Loader paths/indices study", "", "Five paired rounds per cell; matching output pixels and labels.",
             "", f"Decision: **{summary['decision']}**.", "", *limitations, "", "## Warmed time and preparation", "",
             "| Configuration | ns/image median [min, max] | IQR | images/s | startup ns median | list preparation ns median |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, cell in timing["configurations"].items():
        ns = cell["ns_per_image"]
        lines.append(f"| {name} | {ns['median']:.0f} [{ns['min']:.0f}, {ns['max']:.0f}] | {ns['iqr']:.0f} | {cell['images_per_second']['median']:.2f} | {cell['startup_ns']['median']:.0f} | {cell['preparation_ns']['median']:.0f} |")
    lines.extend(["", "## Paired effects and preparation break-even", "", "| Comparison | Median time ratio | Five ratios | Break-even epochs by round | Conclusion |", "|---|---:|---|---|---|"])
    for name, effect in timing["effects"].items():
        conclusion = "material regression" if effect["material_regression"] else ("reliable acceleration in this cell" if effect["reliable_acceleration"] else "no reliable acceleration shown")
        lines.append(f"| {name} | {effect['ratios']['median']:.4f} | {effect['ratios']['values']} | {['no finite measured break-even' if value is None else value for value in effect['break_even_epochs']]} | {conclusion} |")
    lines.extend(["", "## Memory", "", "Input capacity, native path storage, returned arrays, and process RSS/PSS are separate categories. Python manifest lists remain alive in all three variants. Exact per-phase records and parent/worker samples are in memory.jsonl and jobs/*.samples.jsonl.",
                  "", f"Manifest cases: {memory['manifest_cases']}; sequence cases: {memory['sequence_cases']}; retained output control: {memory['output_nbytes_max']} bytes.",
                  "", "| Native checkpoint / repeat | Live native bytes at trace stop | Native bytes at process heap peak |", "|---|---:|---:|"])
    native_heading = lines[-2:]
    del lines[-2:]
    lines.extend(["", "| Manifest case | List ns median | Native construction ns median | Pickle bytes / ns medians | Startup ns median | Parent RSS / PSS medians | Native manifest bytes median |", "|---|---:|---:|---|---:|---|---:|"])
    for name, values in memory["manifest_summary"].items():
        lines.append(f"| {name} | {values['list_ns']['median']:.0f} | {values['construction_ns']['median']:.0f} | {values['pickle_bytes']['median']:.0f} / {values['pickle_ns']['median']:.0f} | {values['startup_ns']['median']:.0f} | {values['rss_bytes']['median']:.0f} / {values['pss_bytes']['median']:.0f} | {values['manifest_bytes']['median']:.0f} |")
    lines.extend(["", "| Sequence / cap / workers / epoch / phase | Parent+workers RSS / PSS | Input retained / peak capacity per Loader | Native handles per Loader | Retained output bytes |", "|---|---|---|---:|---:|"])
    for row in memory["sequence_summary"]:
        lines.append(f"| {row['backend']} / {row['cap']} / {row['workers']} / {row['epoch']} / {row['phase']} | {row['rss_sum']} / {row['pss_sum']} | {row['input_capacity_max']} / {row['input_peak_capacity_max']} | {row['native_live_per_loader_max']} | {row['output_nbytes']} |")
    lines.extend(["", "Complete per-worker manifest observations and per-PID/process-tree sampling peaks are in summary.json. Phase sums cover the parent and active workers; sampled process trees additionally include IPC helpers.", "", *native_heading])
    for row in native:
        lines.append(f"| {row['checkpoint']} / {row['repeat']} | {row['measured']['live']['native_bytes']} | {row['measured']['peak']['native_bytes']} |")
    lines.extend(["", "Live-at-checkpoint allocations intentionally remain alive when heaptrack_stop flushes the trace. Normal Python cleanup follows. The destroyed checkpoint explicitly deletes Loader before trace stop. RSS remaining after frees is allocator/process residency, not exact native retention. Native-at-process-peak is separate from retained native bytes and is not a sum of independent per-stack maxima.",
                  "", "## Reproduction and limitations", "", "See provenance.json for dependencies, wheel paths, CPU/tool identifiers and resolved DataLoader kwargs; study.json for the measurement settings; corpus.tsv and generated-inputs.json for input identities. All raw timings, discarded short attempts, memory samples and complete native stacks are preserved. summary.json contains every paired value and preparation delta, including negative deltas.",
                  "", "These five rounds describe this CPU, corpus and crop/collation pipeline. Direct decode and whole-pipeline effects must be interpreted separately. No universal hardware or training-speed claim follows from this study.", ""])
    (run_dir / "report.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-complete", action="store_true")
    args = parser.parse_args()
    result = generate(load_config(args.run_dir / "study.json"), args.run_dir, args.check_complete)
    print(json.dumps({"complete": result["complete"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()
