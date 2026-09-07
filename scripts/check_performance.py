"""Review benchmark JSON using a paired bootstrap interval; tiny overhead is reported separately."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("result", type=Path)
parser.add_argument("--legacy", type=Path)
args = parser.parse_args()
data = json.loads(args.result.read_text())
rows = {(r["width"], r["height"], r["format"], r["source"], r["profile"]): r for r in data["rows"]}
old = {}
if args.legacy:
    legacy = json.loads(args.legacy.read_text())
    if legacy["python"] != data["python"]:
        raise SystemExit("Baseline and candidate must use the same Python version")
    old = {(r["width"], r["height"], r["format"], r["source"]): r for r in legacy["rows"]}
rng = np.random.default_rng(314)
failed = []
print("| Image | Source | Safe µs | Limits overhead | 95% interval | Old → new |")
print("| --- | --- | ---: | ---: | ---: | ---: |")
for key, row in rows.items():
    if key[-1] != "safe":
        continue
    other = rows[(*key[:-1], "unlimited")]
    a, b = np.array(row["samples_ns"]), np.array(other["samples_ns"])
    # Re-sample matched batches together to retain shared thermal/scheduler drift.
    indices = rng.integers(0, len(a), (10000, len(a)))
    ratios = np.median(a[indices], axis=1) / np.median(b[indices], axis=1)
    low, high = (np.quantile(ratios, [0.025, 0.975]) - 1) * 100
    overhead = (row["median_ns"] / other["median_ns"] - 1) * 100
    previous = old.get(key[:-1])
    change = f'{(row["median_ns"] / previous["median_ns"] - 1) * 100:+.1f}%' if previous else "—"
    print(f'| {key[0]}×{key[1]} {key[2]} | {key[3]} | {row["median_ns"] / 1000:.2f} | {overhead:+.2f}% | [{low:+.2f}%, {high:+.2f}%] | {change} |')
    if key[0] >= 224 and overhead > 2 and low > 2:
        failed.append(key[:-1])
    if key[0] < 224:
        print(f'<!-- Tiny absolute overhead: {(row["median_ns"] - other["median_ns"]) / 1000:+.3f} µs -->')
if failed:
    raise SystemExit(f"Performance gate requires investigation: {failed}")
