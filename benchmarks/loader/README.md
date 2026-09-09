# Loader benchmarks

The benchmark environment adds PyTorch, torchvision and profiling tools; the
runtime imgread package still depends only on NumPy. Run commands from the
repository root. Set the corpus path to your Imagenette validation directory.

```sh
export IMGREAD_CORPUS=/path/to/imagenette2-320/val
export UV_CACHE_DIR=/tmp/imgread-loader-uv-cache
uv sync --project benchmarks/loader --locked --python 3.13.13
uv run --no-sync pytest scripts/tests/test_loader_study.py -q
uv run --project benchmarks/loader --no-sync python benchmarks/loader/run.py --self-check
```

`study.json` contains the matrix and resource settings. The default corpus is
`data/imagenette2-320/val` relative to the repository; `--corpus-root` overrides it.
Selection takes the first 100 JPEGs in sorted filename order from each of ten
sorted class directories. Each run saves the selected paths and sizes in
`corpus.tsv`, along with its configuration and environment.

## Buffer comparison

`buffers.py` compares `load_numpy_from_bytes(data)` with a persistent
`Loader.decode(data)`. All encoded images are loaded before measurement; file
reads are excluded from timed epochs.

The matrix covers `auto` and `image`, immutable `bytes` and `memoryview` (both APIs
copy the latter), direct decode and DataLoader with 0 or 2 workers. The pipeline
uses 224×224 HWC uint8 crops, batch size 32, spawn, persistent workers and no
pinning. Five paired rounds randomize function/Loader order, use equal whole-epoch
counts and target at least two seconds per variant. Full outputs and labels are
hashed outside timing to check that the compared APIs produce the same results.
Worker PIDs must stay stable across epochs. Initialization, validation and warmup
are reported separately. Preloaded buffers are copied into each spawn worker;
that memory and startup cost is separate from decode time.

```sh
uv run --no-sync maturin build --release --locked --out target/buffer-wheels/normal
uv pip install --python benchmarks/loader/.venv/bin/python --no-deps --force-reinstall target/buffer-wheels/normal/*.whl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --project benchmarks/loader --no-sync python benchmarks/loader/buffers.py --corpus-root "$IMGREAD_CORPUS" --output target/buffer-comparison.json
```

Run on Linux with CPUs 0–3 available (or set `--cpus`). Build a normal release
wheel with TurboJPEG; debug and diagnostic builds are rejected for timing.
JSON preserves every round, paired ratios, quartiles, output digests and the
environment. Choose a new output file for each run.

Store benchmark reports and raw results in external project documentation.

## Path/index study

Build a normal release wheel for timing and a diagnostic release wheel with Rust
and native debug symbols for memory profiling. Keep one wheel in each directory.
The supervisor installs the appropriate wheel before each group of stages and
restores the normal wheel after diagnostic stages.

```sh
export IMGREAD_RUN="$(pwd -P)/target/loader-study/run-01"
export IMGREAD_UNIT=imgread-loader-study
uv run --no-sync maturin build --release --locked --out target/loader-wheels/normal
CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_STRIP=none CFLAGS=-g uv run --no-sync maturin build --release --locked --features extension-module,turbojpeg,loader-diagnostics --out target/loader-wheels/diagnostic
uv run --project benchmarks/loader --no-sync python benchmarks/loader/run.py --stage preflight --corpus-root "$IMGREAD_CORPUS" --run-dir "$IMGREAD_RUN"
systemd-run --user --unit="$IMGREAD_UNIT" --property=WorkingDirectory="$(pwd -P)" --property=RuntimeMaxSec=3600 --property=KillMode=control-group --property=TimeoutStopSec=10 --property=MemoryMax=8G --property=TasksMax=64 --property=UMask=0077 --setenv=IMGREAD_UNIT="$IMGREAD_UNIT" --setenv=UV_CACHE_DIR="$UV_CACHE_DIR" --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=RAYON_NUM_THREADS=1 --setenv=CUDA_VISIBLE_DEVICES= uv run --project benchmarks/loader --no-sync python benchmarks/loader/run.py --stage all --run-dir "$IMGREAD_RUN"
```

The output directory can be anywhere writable. Source archives and working trees
with uncommitted edits can run the study. Rebuild the wheels after changing code
and use a new run directory when comparing a different build or configuration.

External prerequisites are Linux x86_64, CPUs 0–3 (configurable in `study.json`),
the isolated Python dependencies, local multiprocessing IPC, systemd user
services, heaptrack and heaptrack_print 1.5.0, 6 GiB available RAM and 10 GiB free
output storage. `--allow-low-memory` records an exception to the RAM admission
threshold; pass it on both preflight and subsequent invocations. The supervisor
stops below 2 GiB free storage or after its one-hour budget. The cgroup limits
apply to the complete process tree.

The stages are `preflight → validate → timing → memory → native → report`.
`--stage` accepts a stage name or `all`.

- Preflight checks the environment and crop dimensions, constructs all twelve
  pipeline configurations, and generates the synthetic inputs.
- Validation compares full pixel/label digests across all eighteen configurations
  outside timed sections. Synthetic baseline JPEG, progressive JPEG and PNG
  fixtures also cover corrupt inputs, resource limits and recovery.
- Timing uses five randomized paired rounds with the same image order and equal
  epoch counts within each cohort. Each accepted round has at least two epochs
  and two seconds of work. Startup, warmup and list preparation are separate.
- Memory measures 1,000-entry and million-entry manifests, ownership variants,
  supported process start methods and three fresh repeats. Small/large/error/
  recovery sequences run for two persistent epochs, with per-worker resource
  diagnostics, an eight-array retention control and process RSS/PSS sampling.
- Native runs eight independent heaptrack prefixes three times. Complete live and
  peak folded stacks are retained. Symbol attribution distinguishes live native
  bytes at a checkpoint from native bytes at the process heap peak.
- Report checks matrix completeness, timing duration, output agreement and native
  symbols, then reports paired ratios, distributions, preparation costs and
  negative results.

`jobs/` holds child requests, results, logs and memory samples; `native/` holds
profiles and stack exports. `control/` inside the run directory holds launch,
heartbeat, stage and completion records. Completed stages can be resumed with
the saved settings when their outputs still exist. A failed or interrupted run
keeps its evidence; start a new directory to repeat it. A local lock prevents
concurrent supervisors from changing the shared benchmark environment.

The optional large-manifest regression, with the diagnostic wheel installed, is:

```sh
IMGREAD_REQUIRE_LARGE_MANIFEST=1 uv run --project benchmarks/loader --no-sync python -m pytest scripts/tests/test_loader_study.py -q -k million_entry
```

## Monitor, cancel and report

```sh
systemctl --user show "$IMGREAD_UNIT" --property=ActiveState --property=SubState --property=ExecMainStatus
journalctl --user --unit="$IMGREAD_UNIT" --no-pager
systemctl --user stop "$IMGREAD_UNIT"
uv run --project benchmarks/loader --no-sync python benchmarks/loader/report.py --run-dir "$IMGREAD_RUN" --check-complete
```

Reports use the settings and results saved in the run directory. They can be
regenerated after editing source code or moving/removing the original corpus.
These measurements describe the selected corpus, CPU and pipeline. Direct decode
and whole-pipeline acceleration are separate questions; no reliable acceleration
is a valid result. RSS residuals are not exact native memory.
