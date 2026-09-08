# Loader study

This isolated CPU environment implements the accepted
`20260908_084136_loader-paths-indices` plan. Runtime imgread still depends only on
NumPy. `study.json` freezes every matrix value, budget, root and decision rule;
`run.py` rejects changed, missing and unknown settings through its canonical hash.
The additional `datasets.py` holds the importable IterableDataset adapter used by
spawn. No runnable imports come from sibling repositories.

## Preparation and code review

Run from the repository root. All Python-facing commands use uv.

```sh
export UV_CACHE_DIR=/tmp/imgread-loader-uv-cache
export PYTHONDONTWRITEBYTECODE=1
uv sync --project benchmarks/loader --locked --python 3.13.13
uv run --no-sync pytest scripts/tests/test_loader_study.py -q
uv run --project benchmarks/loader --no-sync python benchmarks/loader/run.py --self-check
uv run --project benchmarks/loader --no-sync python benchmarks/loader/run.py --help
uv run --project benchmarks/loader --no-sync python benchmarks/loader/report.py --help
```

Before measurements, commit every source, test, config and lock change and obtain
an independent implementation review for the exact clean SHA. Task 7/AC10 remains
pending at that review. The review must be an immutable external sidecar under the
plan's record root, with `Verdict = approved` and `Target commit / HEAD = SHA` in
its metadata table. The harness records its path and full SHA-256.

Build both wheels **after** that approval in otherwise empty normal/diagnostic
build directories. Preflight rejects multiple/stale wheels, verifies ABI/version,
and binds archive and native extension hashes. It checks installed Python helper
bytes against the archive and rejects editable/sibling origins. A private native build stamp records the Git SHA, clean/dirty state, release profile
and diagnostic debug flags. Every child verifies this stamp against the approved
SHA in addition to archive hashes; no local paths are embedded in the stamp.
Source archives without Git metadata cannot masquerade as reviewed study builds.

```sh
export IMGREAD_SHA="$(git rev-parse HEAD)"
export IMGREAD_RUN="/home/aya/Prj/PrjDocs/imgread/docs/experiments/20260908_084136_loader-paths-indices/$IMGREAD_SHA/attempt-01"
export IMGREAD_UNIT="imgread-loader-study-$IMGREAD_SHA-a01"
umask 077
uv run --no-sync maturin build --release --locked --out target/loader-wheels/normal
CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_STRIP=none CFLAGS=-g uv run --no-sync maturin build --release --locked --features extension-module,turbojpeg,loader-diagnostics --out target/loader-wheels/diagnostic
uv pip install --python benchmarks/loader/.venv/bin/python --no-deps --force-reinstall target/loader-wheels/normal/*.whl
uv run --project benchmarks/loader --no-sync python benchmarks/loader/run.py --config benchmarks/loader/study.json --stage preflight --run-dir "$IMGREAD_RUN"
systemd-run --user --unit="$IMGREAD_UNIT" --property=WorkingDirectory="$(pwd -P)" --property=RuntimeMaxSec=3600 --property=KillMode=control-group --property=TimeoutStopSec=10 --property=MemoryMax=8G --property=TasksMax=64 --property=UMask=0077 --setenv=UV_CACHE_DIR=/tmp/imgread-loader-uv-cache --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=RAYON_NUM_THREADS=1 --setenv=CUDA_VISIBLE_DEVICES= uv run --project benchmarks/loader --no-sync python benchmarks/loader/run.py --config benchmarks/loader/study.json --stage all --run-dir "$IMGREAD_RUN"
```

External prerequisites: Linux x86_64, CPUs 0–3 available, CPython 3.13.13, the
committed isolated lock, local multiprocessing IPC, systemd user services,
heaptrack **and heaptrack_print 1.5.0**, at least 6 GiB available RAM and 10 GiB free
output storage. Profiler provisioning is external; this harness does not install
system packages. It stops below 2 GiB free storage or after its attempt budget.
No network, GPU, uploads or writes to original image data are part of a study.

An owner-authorized run on a memory-constrained host may add `--allow-low-memory`
to **both** the preflight and full supervisor commands. Save the owner's decision
in the external launch record. This waives only the 6 GiB admission check for that
attempt; the frozen matrix, correctness checks, cgroup and time limits stay intact.
The flag is part of the attempt binding, so it cannot change when resuming stages.
Provenance records actual available memory; the report labels the exception and
requires confirmation on another machine. The default remains strict.

## Stages and evidence

The DAG is `preflight → validate → timing → memory → native → report`. `--stage`
accepts each name or `all`. Stage selection never changes scientific settings.
Completed stages can be reused only when every artifact hash and all provenance
match. Interrupted timings restart completely; incomplete rows and child logs
remain available. Re-executed child requests get distinct resume suffixes.

- Preflight constructs all twelve pipeline DataLoader configurations using the
  timing factory's explicit worker profiles. It validates versions, wheel origin,
  corpus digest and crop dimensions and writes provenance and synthetic recipes.
- Validation compares full pixel/label digests outside timed sections for all
  eighteen configurations, then checks deterministic small/large/progressive,
  corrupt and oversized fixtures. The corrupt JPEG has exactly four bytes at
  `entropy_start+16` replaced by `FF C4 00 01`; no on-run mutation search is allowed.
  Each of the 64×64 and 2048×2048 RGB arrays uses a fresh PCG64(37), with baseline
  JPEG, progressive JPEG and PNG variants. The limit fixture starts from a 2×2
  black JPEG. Generator/Pillow versions and every file's size/hash are recorded.
- Timing uses five randomized rounds, a common seed-37 image order and equal epoch
  counts within each function/path/index cohort. Each accepted round has at least
  two whole epochs and two seconds of work. Warmup, startup and list preparation
  are separate from elapsed decode/collation times. Short calibration attempts are
  retained separately and never count as results.
- Memory measures both manifest lengths, all three ownership variants, supported
  process start methods and three fresh repeats. Sequence workers each perform
  every phase independently for two persistent epochs, with parent barriers and
  exact resource diagnostics. A separate eight-array retention control reports
  output nbytes, followed by array and Loader deletion. Process RSS/PSS/peak RSS
  sampling occurs only in memory jobs, every 100 ms.
  Manifest startup and phase handshakes share the configuration's 120-second
  deadline, including the time spent spawning/unpickling the other worker.
  The explicit pre-review regression is
  `IMGREAD_REQUIRE_LARGE_MANIFEST=1 uv run --project benchmarks/loader --no-sync python -m pytest scripts/tests/test_loader_study.py -q -k million_entry`
  with a diagnostic wheel installed; it checks the million-entry indexed spawn
  case without creating scientific timing or memory-study results.
- Native runs all eight independent heaptrack prefixes three times. A symbolized
  diagnostic wheel supplies the prefixes; the normal wheel supplies all timings.
  The child calls `heaptrack_stop()` while the checkpoint is alive, then permits
  normal Python cleanup. Live native bytes and native bytes at the process heap
  peak are distinct. Full folded stacks are retained; each allocation is assigned
  once to its nearest native allocator frame. heaptrack 1.5.0 assigns allocation
  peak costs at the global heap peak, as shown in its
  [trace reader](https://github.com/KDE/heaptrack/blob/v1.5.0/src/analyze/accumulatedtracedata.cpp).
- Report rejects missing cells, short/unequal rounds, mismatched hashes or absent
  native symbols. It reports all paired ratios, median/IQR/min/max, startup/list
  costs, break-even and negative results. A material regression returns to the
  owner before any optimization claim.

`timings.jsonl` stores config ID, round/epoch, integer elapsed nanoseconds, sample
count, order/output digests, startup/preparation times and full binding.
`memory.jsonl` stores scenario, phase/epoch, parent/worker PID and role, RSS/PSS,
peak RSS, diagnostics and retained output bytes. `jobs/` preserves requests,
results, stdout/stderr and sampled process observations. Native compressed traces,
text output and complete live/peak stack exports live in `native/`.

The output directory is private (0700; files 0600). Mutable launch/heartbeat/stage
and completion records remain in the distinct external workflow control directory
`docs/plans/<id>.<SHA>.attempt-01/`. An exclusive local file lock allows only one
active attempt. One documented infrastructure/interruption/resource retry is
allowed as `attempt-02`; semantic failures require a code fix and new SHA/review.
An attempt with a terminal failure cannot be reopened under its original number.
Within an attempt, staged invocations resume only at a successfully recorded
stage boundary. A killed/in-progress stage requires a documented failure and the
permitted next attempt; its calibration allowance cannot reset in place.
No timing selection or corpus/matrix reduction is allowed to obtain a positive
result.

## Monitor, cancel and check completeness

```sh
systemctl --user show "$IMGREAD_UNIT" --property=ActiveState --property=SubState --property=ExecMainStatus
journalctl --user --unit="$IMGREAD_UNIT" --no-pager
uv run --project benchmarks/loader --no-sync python benchmarks/loader/report.py --run-dir "$IMGREAD_RUN" --check-complete
```

Monitor phase/state every 30 seconds. A heartbeat older than 60 seconds is failure
and requires stopping the service. To cancel, explicitly run:

```sh
systemctl --user stop "$IMGREAD_UNIT"
```

The service terminates its control group after the ten-second grace period. Child
jobs also run in killable process groups. No killed/interrupted stage may be marked
successful. The supervisor restores the normal wheel in `finally` after diagnostic
stages, without replacing a library inside a live interpreter that imported it.

A complete report is local evidence for this corpus, CPU and pipeline. Direct
and whole-pipeline acceleration are separate questions. No reliable acceleration
is a valid result; RSS residuals are never reported as exact native memory.
