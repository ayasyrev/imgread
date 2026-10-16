# Direct NumPy-owned JPEG output

The user explicitly requested implementing the discussed NumPy-owned output
scheme and checking it in a separate worktree (2026-10-03). This is the accepted
simple implementation task; formal task/plan reviews are not applicable.

- Worktree: `/home/albu/Prj/AYA/imgread/worktree/numpy-owned-output-20261003`.
- Branch: `aya/numpy-owned-output-20261003`.
- Base: `d1ae4dd` (current committed checkout, including the existing memchr scanner).
- Scope: Python native JPEG fast path for functional APIs and Loader uses
  NumPy allocation before TurboJPEG writes pixels. Preserve RGB/BGR uint8,
  fresh independently owned outputs, limits, fallback warnings/errors, and
  process/reentry/reuse behavior. Rust image-returning APIs and image-backend
  fallback keep their existing storage contracts.
- Verification: actual NumPy ownership and allocator callbacks, error cleanup,
  retained views, existing Rust/Python decoder and Loader checks, then matched
  release-build timing on identical JPEG bytes. No publication or merge implied.

## Implementation

`src/output.rs` separates native output storage from decoding policy. The Python
JPEG path calls `PyArray_Empty` for a C-contiguous `(height, width, 3)` uint8 array
after header/size/budget checks. TurboJPEG writes directly to its data pointer
without the GIL. The array is private until decoding succeeds; errors discard
it, and the existing fallback policy still runs. The returned array has
`OWNDATA=True` and `base is None`. There is no Rust Vec, ImageBuffer, ndarray or
PySliceContainer on this successful Python native JPEG output path.

The Rust image API still allocates a Vec through the same generic decoding
policy. Fallbacks still transfer the image backend's Vec into NumPy. This is a
change of allocator and ownership, not removal of an RGB memcpy: the original
native path already transferred its pixel allocation without copying it.

NumPy allocation reacquires the GIL between native header parsing and entropy
decoding. Loader temporarily moves its workspace out of the mutex during that
operation; its existing admission guard continues to reject reentry. A strong
Python reference owns the allocation while native code writes without the GIL.
Allocation errors remain terminal resource errors rather than fallback triggers.

## Validation

- Native release wheel, Python 3.12.13 / NumPy 2.5.3: full Python suite
  **595 passed, 8 skipped**. Two additional native-error/fallback cleanup cases
  were then added; the complete new output suite passes **18 tests** on the final
  release wheel. The subsequent source adjustment only handles destructuring in
  builds without the TurboJPEG feature.
- Diagnostic native wheel: all new output tests plus Loader resource/reuse tests,
  **24 passed**. Tests exercise all Python entry points, RGB/BGR, baseline and
  progressive JPEG, grayscale, retained views, independent mutation, actual
  NumPy allocation/free, no calloc, injected allocation failure, native failure
  before fallback, and allocator callback reentry.
- Build without TurboJPEG: full Python suite **579 passed, 26 skipped**;
  expected native-unavailable fallback warnings remain.
- `cargo fmt -- --check`, Clippy all targets with default and all features,
  `cargo test --locked` and `cargo test --locked --features turbojpeg` pass.
  Default features are empty, so a separate no-default-features Clippy run would
  be identical. Ruff passes for both new Python files.
- Final native wheel with the minimum supported NumPy **1.26.4**: all **18**
  output ownership/allocator/error tests pass as well.
- Tested Linux x86_64. The full Python/platform release matrix was not run.

## Matched benchmark

The original is a clean `git archive d1ae4dd2b3ea554c39856993c698a1e999f7730a`;
the candidate is this worktree's implementation. Both are release CPython 3.12
manylinux2014 wheels, built in the same saved container
`localhost/imgread-manylinux-followup:20261002` with Rust 1.95.0, GCC 10.2.1,
Maturin 1.11.5 and vendored static libjpeg-turbo 3.1.0. Native C uses `-O3`, SIMD
enabled and no C compilation LTO. Identical final linker flags in both builds:

```text
RUSTFLAGS=--remap-path-prefix=/io=. --remap-path-prefix=/build=. -C link-arg=-fuse-ld=bfd -C link-arg=-flto
CFLAGS=-ffile-prefix-map=/io=. -ffile-prefix-map=/build=.
TURBOJPEG_STATIC=1
TURBOJPEG_SOURCE=vendor
```

Compiler dependencies and C objects are shared between these sequential builds;
the wheel output directories and installed modules are separate. Production
wheels, without `loader-diagnostics`, are timed. Simplejpeg is the installed
1.9.0 wheel, whose native build differs; it is a parity reference, while the two
imgread wheels isolate this implementation change.

The corpus is the existing 128-image Imagenette RGB uint8 256x256 prepared
subset from the RAM-codecs experiment. Its immutable JPEG buffers were saved
under `/tmp/imgread-release-followup/buffers`: Pillow 12.3.0 linked to
libjpeg-turbo 3.1.4.1 encodes Q50, Q90, Q100 baseline and Q90 progressive, all
4:2:0. Every imgread route is checked
for exact pixel equality with `simplejpeg.decode_jpeg(..., colorspace="RGB",
fastdct=False, fastupsample=False)` on all 512 buffers before that case's timing.
Candidate ownership and baseline external ownership are also asserted.

Host: AMD Ryzen 7 5800X. All timed variants are pinned to CPU 3. A separate
experiment occupied CPU 7; its work is not included in these timings. Each of
10 blocks uses a shuffled image order and a shuffled forward/reverse order of
baseline, candidate and simplejpeg. Each sample makes five full corpus passes.
The timer includes the public call, allocation, decoding, returned NumPy result
and disposal. Encoding, I/O, imports, correctness checks and warmup are excluded;
GC is disabled only inside each timed sample. Thread-library limits are one.

Raw per-sample wall/CPU timings and command are committed in
[`paired-256.json`](../../experiments/numpy_output/results/paired-256.json).
The unexpectedly larger Loader benefit on progressive JPEG warranted a fresh
process with candidate-first module loading and validation; its raw record is
[`paired-256-candidate-first.json`](../../experiments/numpy_output/results/paired-256-candidate-first.json).
The confirmation changes the order of initial native-library/output allocations
and the balanced timing order, while preserving inputs, flags and endpoints.
The independent runner is [`compare.py`](../../experiments/numpy_output/compare.py):

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/imgread-diagnosis-uv-cache \
uv run --no-project \
  --python /home/albu/Prj/AYA/imgread_benchmark/worktree/ram-image-codecs/experiments/ram_image_codecs/.venv/bin/python \
  python experiments/numpy_output/compare.py \
  --baseline-dir /tmp/imgread-numpy-output-20261003/install-baseline \
  --candidate-dir /tmp/imgread-numpy-output-20261003/install-candidate \
  --buffers /tmp/imgread-release-followup/buffers \
  --output experiments/numpy_output/results/paired-256.json \
  --cpu 3 --blocks 10 --repeat 5
# Confirmation: add --candidate-first and a different --output filename.
```

## Results

Positive saved percentages mean lower decode time than the original imgread.
Every percentage is the median of ten paired block ratios, not a ratio of
unpaired aggregate medians. The simplejpeg column is the candidate's excess
time in the confirmation run; negative means the candidate is faster.

| JPEG 4:2:0 | API | Time saved, first run | Time saved, confirmation | Candidate vs simplejpeg, confirmation | Faster than original, confirmation |
|---|---|---:|---:|---:|---:|
| Q50 baseline | functional | +0.08% | +0.75% | +9.60% | 9/10 |
| Q50 baseline | simple | -0.02% | +0.55% | +9.11% | 10/10 |
| Q50 baseline | Loader | +1.93% | +0.38% | +1.31% | 10/10 |
| Q90 baseline | functional | +0.39% | +1.01% | +4.75% | 10/10 |
| Q90 baseline | simple | +0.35% | +0.73% | +4.66% | 10/10 |
| Q90 baseline | Loader | +1.55% | +0.61% | +0.01% | 9/10 |
| Q100 baseline | functional | +1.30% | +1.59% | -0.11% | 10/10 |
| Q100 baseline | simple | +1.22% | +1.51% | -0.88% | 10/10 |
| Q100 baseline | Loader | +2.10% | +1.30% | +0.32% | 10/10 |
| Q90 progressive | functional | -0.01% | -0.01% | +0.16% | 5/10 |
| Q90 progressive | simple | -0.57% | -0.00% | -0.06% | 4/10 |
| Q90 progressive | Loader | +4.36% | +0.16% | -0.01% | 9/10 |

The output scheme works and removes the external owner. This implementation
shows a small benefit on baseline JPEG, most consistently at Q100. Its magnitude
depends on the process/run: the initial 4.36% progressive Loader benefit shrinks
to 0.16% in confirmation. Do not claim that 4% as a reproducible optimization,
or attribute it specifically to removing the 48-byte owner. Loading/allocation
order and other run-to-run effects remain possible explanations; this experiment
does not separate them. The tiny progressive differences are not evidence of a
useful speedup or of exact speed equality.

The scheme does not establish parity with simplejpeg for all public functions:
the Q50/Q90 baseline functional routes still take about 5-10% longer in the
confirmation. Loader is much closer: effectively tied in these Q90 measurements,
about 1.3% slower at Q50, and 0.3-1.4% slower at Q100 across the two runs.
Functional APIs still copy the compressed buffer and create a fresh native
decoder; Loader already borrows immutable bytes and reuses certified native
state. Those differences were preserved and must not be credited to the output
allocation change.

This is evidence for the committed implementation on 128 prepared images and
one CPU, not a general performance guarantee or a full release qualification.
The Python JPEG path also retains the extra GIL reacquisition for NumPy
allocation; the experiment does not isolate the cost of that transition.
