# jpegli cache encoding and CPU decoding

`imgread.encode_jpeg` prepares ordinary JPEG bytes using the optional ajpegli
encoder. Keep one `Loader` per worker to decode immutable cached bytes. The
encoder defaults to sequential JPEG, quality 90 and 4:2:0 subsampling.

The implementation also replaces the scalar entropy-byte loop in Loader's
table-independence certificate with `memchr`. It still inspects every marker,
including tables between progressive scans. Decode limits, DCT, upsampling,
fallbacks and output ownership are unchanged.

## Measurement, 2026-10-02

[Raw measurements](results/20261002-cpu7.json), including every timed pass,
paired speed ratios, size, encoding time and reconstruction error.

- 2,000 prepared Imagenette images, RGB uint8, 256 x 256.
- AMD Ryzen 7 5800X; one process pinned to logical CPU 7; output in CPU RAM.
- Python 3.12.13, NumPy 2.5.2, ajpegli 1.0.0, simplejpeg 1.9.0.
- jpegli revision `7cdf212790241868c77dca777dbee14e98128cba`.
- Baseline imgread: `2096de5`; baseline and modified extensions both built in
  release mode with TurboJPEG on the same host/toolchain.
- Each setting is encoded once. All five reader routes receive the same immutable
  byte objects. No file access, encoding, resize, validation or warmup is timed.
- Timing includes RAM lookup, the direct public call, materialization of a new
  writable contiguous RGB array and output disposal. No output buffers are reused.
- Five rounds, each with a shuffled image order and both forward and reversed
  reader orders (AB/BA); ten observations per reader/setting, 700,000 timed decodes
  in total. Warnings are errors, with the filter set outside the timed loop.
- Every image is checked across current/baseline Loader, current/baseline
  `load_numpy_from_bytes` and simplejpeg. Maximum pixel difference was **zero**
  for every setting. Output independence is also checked outside timing.

Throughput below is the median images/s. PSNR is computed from aggregate MSE over
all source pixels, using the current Loader's decoded pixels. Equal quality
numbers do not imply equal distortion across encoders.

| Encoder and setting | KiB/image | PSNR dB | Previous Loader | New Loader | simplejpeg |
| --- | ---: | ---: | ---: | ---: | ---: |
| jpegli Q50, 420, sequential | 6.61 | 30.27 | 8,000 | 8,059 | 8,164 |
| jpegli Q75, 420, sequential | 10.08 | 32.83 | 7,545 | 7,645 | 7,765 |
| jpegli Q90, 420, sequential | 17.04 | 35.88 | 5,376 | 5,422 | 5,561 |
| jpegli Q95, 420, sequential | 24.75 | 37.63 | 4,357 | 4,444 | 4,527 |
| jpegli Q90, 420, progressive level 2 | 16.67 | 35.88 | 2,119 | 2,118 | 2,028 |
| jpegli Q90, 444, sequential | 19.36 | 37.27 | 3,952 | 3,998 | 4,002 |
| simplejpeg Q90, 420, sequential | 20.05 | 35.52 | 4,768 | 4,847 | 4,959 |

The marker-search change gives median paired improvements of **0.82–1.73%** on
the four sequential jpegli 420 settings; all five paired rounds favor the new
Loader on each of these settings. This is a small improvement. simplejpeg still
leads by approximately **1.3–2.3%** in the paired measurements. No Loader
improvement is established for progressive JPEG. On that stream, imgread is
approximately 4.4% faster than simplejpeg already in the baseline build.

The 444 comparison does **not establish a difference** between imgread and
simplejpeg: the median gap is tiny and two rounds show substantial timing
disturbance. Some Q50 observations also ran at a different absolute rate;
the paired speed ratios, not cross-pass extrema, support its small improvement.
All individual observations are retained in the JSON.

The more consequential cache choice is sequential encoding: at jpegli Q90/420,
it decoded **2.56x faster** than progressive level 2 with **2.18% more bytes**
and the same measured aggregate PSNR. At numeric Q90/420, jpegli used **15.0%
fewer bytes** than simplejpeg's encoder, decoded **11.8% faster** through the new
Loader and had PSNR 35.88 versus 35.52 dB. This is not a matched-distortion study
or evidence of unchanged training accuracy.

The plain imgread bytes function remains competitive: on jpegli Q90/420 it
measured 5,445 images/s (baseline 5,462). These are direct calls. The older
128-image comparison in imgread_benchmark wrapped each imgread call in
`warnings.catch_warnings()` and called `load_numpy_from_bytes`, so it does not
isolate Loader performance or the marker-search change.

## Reproduce

The benchmark takes a prepared `N,H,W,3` uint8 `.npy` array, so dataset preparation
is separate from this timing. It needs the optional encoder and simplejpeg:

```sh
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python --no-deps ajpegli==1.0.0 simplejpeg==1.9.0
UV_CACHE_DIR=.uv-cache uv run --no-sync maturin develop --release --locked
UV_CACHE_DIR=.uv-cache uv run --no-sync python experiments/jpegli_cache/benchmark.py \
  --prepared /path/to/prepared.npy --cpu 7 --count 2000 --rounds 5 \
  --output /path/to/new-report.json
```

The recorded corpus was
`../imgread_benchmark/worktree/ram-image-codecs/results/ram-codecs-20261001-simple/prepared.npy`.
For before/after comparisons, first build a wheel from the baseline commit in
a separate checkout, extract it, and add
`--baseline-native /path/to/extracted/imgread/_native.cpython-312-x86_64-linux-gnu.so
--baseline-label 2096de5` to the command. The script loads that extension under a
separate module name, so both implementations are timed in the same process.
Use the same Python ABI, build profile, compiler and native features for both.
A 32-image smoke used the same source, versions and settings before the full run.

## Validation

- Python: `uv run --no-sync pytest python_tests -q`: **567 passed, 9 skipped**
  in the imgread environment (optional Torch and diagnostic-only cases skipped).
- Rust: `cargo test --locked --features turbojpeg`: **42 passed**;
  `cargo test --locked`: **35 passed**.
- `cargo fmt -- --check`, all-feature Clippy with warnings denied, and
  `git diff --check` passed.
- A release wheel and source distribution were built locally. The optional
  encoder remains separate from the native decoder wheel.

The sandbox blocks local sockets required by multiprocessing tests; the passing
Python run used the normal host environment and imgread's own installed metadata.

These measurements cover single-threaded RAM decoding at 256 x 256. They do not
measure file/shard I/O, DataLoader throughput or model quality. Bigger images,
other CPUs and multiple workers need their own workload measurement.
