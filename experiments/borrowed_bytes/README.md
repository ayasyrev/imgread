# Borrow immutable input in the public bytes functions

Accepted 2026-10-04 in this chat: “давай попробуем - делаем в отдельной ворктри”,
following the proposed bytes borrowing, common-native-core comparison, and
conditional investigation of a remaining gap. This is a simple accepted task;
formal task/plan review is not applicable. No publication or merge is requested.

- Base: `ee98d996d0388bef66f0182c4acfae20a146c81c`, NumPy-owned JPEG output.
- Branch: `aya/borrowed-bytes-20261004`.
- Worktree: `/home/albu/Prj/AYA/imgread/worktree/borrowed-bytes-20261004`.

Plan: change only encoded-input ownership in both public functional bytes APIs,
check input limits/lifetime and snapshots of mutable/read-only/strided exporters,
then compare release baseline/candidate and simplejpeg after a passing smoke.
The ordinary fresh writable RGB output and disposal stay inside the timer.
Exact pixels, ownership, warning/fallback behavior and existing buffer contracts
must be preserved. Speed is an experimental result, not an assumed acceptance
criterion for removing an unnecessary copy.

The shared helper now returns a borrowed `Cow` for `PyBytes` and an owned snapshot
for every other buffer. The Python argument remains alive while native code runs
without the GIL, including output allocator callbacks. Loader already borrows
bytes and retains that path. Native handle policy, decoder flags, safe limits,
output allocation and release packaging are unchanged.

## Evidence discovered since the source comparison

Existing committed sibling experiments already compared a common native core
and identified an address-dependent native store penalty. Their work is reused
as evidence, rather than rerun as if it were new:

- `worktree/jpeg-common-core-diagnosis-20261004`, commit `142edb7`:
  `experiments/common_core/README.md` and `experiments/variable_sizes/README.md`.
  The fixed-address control explains a possible 5–10% gap, without assigning
  that cause to uninstrumented historical timings. Varied-size public calls
  did not establish a universal 5–10% imgread penalty.
- `worktree/simplejpeg-hypotheses-20261004`, committed `cfc9219` / `761163b`:
  same-binary copy/borrow and libturbojpeg 3.1.2 controls. Private borrow/shared
  results cannot be called public-API results. This worktree implements the
  corresponding small public change and checks its actual compiled functions.

Primary comparisons use matched builds and paired forward/reverse blocks in
fresh processes. Common-core comparison reuses an existing unmodified shared
libturbojpeg and its matching simplejpeg build, verifies actual runtime linkage,
and measures the new public function. Existing mutable work in sibling trees
must not be changed. Fixed-size allocation/order sensitivity is recorded as a
limitation; conditional address control is appropriate if a large gap reappears.

## Result

Both public bytes functions now avoid allocating and copying immutable JPEG
input. The new public function is close to simplejpeg when they use the same
native library. Removing the copy does **not** demonstrate a universal 4–7%
speedup: most of that fixed256 improvement disappears when all routes receive
the same RGB address class. On original differently sized JPEGs, the measured
improvement over the unchanged function is 0.48–0.65% in the two runs.

Keep the small ownership change: it removes one unnecessary allocation/copy
without changing output or mutable-buffer semantics. These results do not
justify rewriting the binding, changing decoder flags, or changing the product
allocator. Distribution packaging remains a separate performance variable.

### Protocol and builds

- Linux x86_64, AMD Ryzen 7 5800X, pinned logical CPU3; Python 3.12.13,
  NumPy 2.5.3, simplejpeg 1.9.0, Rust 1.97.1 and GCC 15.2.
- Baseline and candidate are **local release extensions**, not portable
  manylinux wheels. They differ only by the `src/lib.rs` patch embedded in
  each raw result's build record. The builder cleans the imgread package
  between variants and checks the compiled function docstring to rule out
  Cargo incorrectly reusing the same-name baseline artifact.
- Static pair: vendored libjpeg-turbo 3.1.0, compared with the installed
  simplejpeg wheel and its separate native build. Common pair: both imgread
  variants and rebuilt unmodified simplejpeg use the **same original** shared
  libjpeg-turbo 3.1.0. No temporal-store patch is used. `ldd` and runtime
  mappings confirm linkage; see [native build provenance](results/shared-native-build.json).
- Fixed corpus: 128 images per Q50/Q90/Q100 sequential and Q90 progressive,
  256×256 RGB, 4:2:0; the existing `/tmp/imgread-release-followup/buffers` files
  are reused without re-encoding. These are 512 JPEG variants, not 512
  independent source photographs.
- Native corpus: 64 original JPEGs, 37 H/W shapes, 33 decoded byte sizes,
  51 × 4:4:4, 12 × 4:2:0 and one grayscale; all sequential. Original geometry
  and bytes are preserved. See [the corpus manifest subset](results/native-corpus.json).
- Every route/input must match simplejpeg RGB exactly and return uint8,
  writable contiguous independent storage, `OWNDATA=True`, `base is None`.
  Validation arrays are released before timing. Input file I/O, validation,
  warmup and result serialization are outside the timer.
- Seven directly called public routes: baseline/candidate generic function,
  simple function, Loader, and simplejpeg. Timers include RAM lookup, decode,
  fresh output allocation and disposal. No output is reused.
- Ten paired blocks, each with shuffled method/image order followed by the
  reverse method order, two image passes per sample. A second fresh process
  changes module load order and shuffle seed. Wall time, process CPU time and
  minor faults are retained for every sample. All our runs were serialized.

All percentages below are `100 * (candidate time / comparator time - 1)`;
**negative means faster**. Each cell is the paired wall-time median in the
first / second process. These are observed medians, not confidence intervals.

### Ordinary allocation, common native core

| 256×256 input | New function / previous function | New function / simplejpeg |
| --- | ---: | ---: |
| Q50 sequential | -7.20% / -6.94% | -0.92% / +0.07% |
| Q90 sequential | -4.53% / -5.06% | +0.59% / -0.06% |
| Q100 sequential | -0.42% / -0.50% | +0.05% / +0.54% |
| Q90 progressive | +0.31% / +0.68% | -0.20% / -0.42% |

All ten blocks favor the new function on Q50 and Q90 in both processes.
The candidate is within ±1% of simplejpeg in each of these eight median
comparisons. Progressive is slightly slower than the previous function;
this change does not accelerate every JPEG case.

### Fixed RGB address class, common native core

The existing private NumPy policy gives every fresh output base address
`mod64=16`, checked during validation for every route/input. At RGB pitch768,
this avoids the native 32-byte-aligned store branch on every row. It adds the
same allocator logic to all routes and preserves output ownership; it is a
diagnostic, not a production change.

| 256×256 input | New function / previous function | New function / simplejpeg |
| --- | ---: | ---: |
| Q50 sequential | -0.40% / -1.30% | -0.88% / -0.54% |
| Q90 sequential | -0.20% / +0.04% | -0.37% / +1.11% |
| Q100 sequential | -0.02% / -0.52% | +0.51% / +0.14% |
| Q90 progressive | +0.11% / -0.41% | +0.08% / +0.07% |

The original 4–7% difference largely disappears. This is consistent with the
previously established address-dependent native store mechanism and allocation
state changing when the encoded-input allocation is removed. It is **not** a
measurement of pure memcpy cost: this experiment does not record the pointers
inside ordinary-allocation timed loops or isolate every native scratch
allocation. It cannot assign a native branch to each earlier sample.

### Original JPEG sizes, ordinary allocation, common native core

| 64 original JPEGs | First process | Second process |
| --- | ---: | ---: |
| New function / previous function | -0.48% | -0.65% |
| New function / simplejpeg | -0.27% | -0.47% |
| New simple function / previous simple function | -0.45% | -0.58% |
| New simple function / simplejpeg | -0.61% | -0.95% |

No persistent large wrapper penalty appears on this corpus. The generic
function's block ranges against its predecessor still cross zero:
-2.02..+0.77% and -1.63..+0.76%. The small median gain is not a universal
per-image guarantee. Loader is an unchanged-path control; its candidate/baseline
medians here are +0.16% and -0.18%.

### Separate native builds

| 256×256 input | New static function / previous static function | New static function / installed simplejpeg |
| --- | ---: | ---: |
| Q50 sequential | -6.43% / -7.68% | +2.69% / +1.15% |
| Q90 sequential | -4.01% / -4.19% | +1.22% / +1.81% |
| Q100 sequential | +0.12% / +0.45% | +3.37% / +1.69% |
| Q90 progressive | +0.41% / +1.06% | -5.18% / -4.78% |

This comparison still includes native version/build differences and the
default allocator. It does not establish installed-wheel parity or isolate
which build flag causes the residual. The common-core result is the appropriate
control for the wrapper comparison.

### Noise and scope

This is a shared workstation, not an isolated benchmark host. Another Python
benchmark on CPU7 was observed during the initial runs; it was not stopped or
modified. Some static-forward samples have wall/CPU ratios around2 and large
outliers (Q100 previous-function comparison reaches -32.77%). All samples are
retained. Process-CPU medians retain the main Q50/Q90 directions; the common-core
default runs have maximum wall/CPU ratios below1.03. Smaller outliers also occur
in address controls. Do not compare absolute times across separate processes
as if CPU availability/frequency were held fixed.

Full blocks, paired ranges and slower-block counts are in the raw JSON files;
[summary.json](results/summary.json) also derives CPU-time comparisons. Smoke
records establish execution/correctness only and are excluded from this table.
No training, multiworker, file-I/O, other-CPU or release-wheel claim is made.

## Validation

- Static candidate focused Python suite: **304 passed, 2 skipped**.
- Common-core candidate buffer/output/simple/limit suite: **266 passed,
  2 skipped**. Skips require the separate diagnostic Loader feature.
- Rust with `turbojpeg`: **45 passed**; default features: **38 passed**.
- Clippy release/all targets/all features with warnings denied, Rust format,
  Python Ruff and `git diff --check`: passed.
- Exact RGB/ownership checks passed for all 512 fixed-corpus JPEGs and all
  64 native originals in every executed route/process/policy.

New tests check original pointer identity for bytes, input limits, bytearray
snapshot independence, temporary bytes-subclass lifetime across NumPy allocator
callbacks, and mutation of bytearray/read-only/strided backing storage from
those callbacks. Other exporters continue to copy before releasing the GIL.

## Reproduction

Run from this worktree. Python must contain NumPy, simplejpeg, Pillow and pytest;
Cargo dependencies, CMake, NASM and Ruff must be available. Existing shared-core,
alignment and corpus assets come from sibling commit `142edb7`:
`experiments/common_core/build_shared.py`,
`experiments/variable_sizes/build_policy.py` and
`experiments/variable_sizes/prepare.py`. Their source/recipe metadata is retained
above; this task reuses their original unmodified binaries and corpus.

The commands below choose new build/output directories. Change the paths if
those directories already exist. They do not install over an existing package.

```bash
BENCH_PY=/home/albu/Prj/AYA/imgread_benchmark/worktree/ram-image-codecs/experiments/ram_image_codecs/.venv/bin/python
COMMON=/tmp/imgread-common-core-20261004
STATIC=/tmp/imgread-borrow-replay-static
SHARED=/tmp/imgread-borrow-replay-common
FIXED=/tmp/imgread-release-followup/buffers
POLICY=/tmp/imgread-variable-sizes-20261004/alignment-policy/_imgread_alignment_policy.cpython-312-x86_64-linux-gnu.so
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$COMMON/native/lib"

"$BENCH_PY" experiments/borrowed_bytes/build_pair.py --output "$STATIC" \
  --cmake-dir /home/albu/Prj/AYA/imgread/.venv/bin \
  --nasm /home/albu/Prj/AYA/imgread/target/loader-toolchain/nasm/usr/bin/nasm
"$BENCH_PY" experiments/borrowed_bytes/build_pair.py --output "$SHARED" \
  --native-prefix "$COMMON/native" \
  --cmake-dir /home/albu/Prj/AYA/imgread/.venv/bin \
  --nasm /home/albu/Prj/AYA/imgread/target/loader-toolchain/nasm/usr/bin/nasm

shared_args=(--build "$SHARED"
  --simplejpeg-extension "$COMMON/simplejpeg-source/simplejpeg/_jpeg.cpython-312-x86_64-linux-gnu.so"
  --native-library "$COMMON/native/lib/libturbojpeg.so.0")

for mode in smoke forward reverse; do
  run_args=()
  if [ "$mode" = smoke ]; then run_args=(--count 8 --blocks 1 --repeat 1); fi
  if [ "$mode" = reverse ]; then run_args=(--candidate-first); fi
  "$BENCH_PY" experiments/borrowed_bytes/compare.py --build "$STATIC" \
    --buffers "$FIXED" "${run_args[@]}" \
    --output "/tmp/imgread-borrow-replay-results/static-$mode.json"
  "$BENCH_PY" experiments/borrowed_bytes/compare.py "${shared_args[@]}" \
    --buffers "$FIXED" "${run_args[@]}" \
    --output "/tmp/imgread-borrow-replay-results/common-$mode.json"
  "$BENCH_PY" experiments/borrowed_bytes/compare.py "${shared_args[@]}" \
    --buffers "$FIXED" --alignment-module "$POLICY" --offset 16 "${run_args[@]}" \
    --output "/tmp/imgread-borrow-replay-results/aligned-$mode.json"
  "$BENCH_PY" experiments/borrowed_bytes/compare.py "${shared_args[@]}" \
    --buffers /tmp/imgread-variable-sizes-20261004/corpus --cases native-original \
    "${run_args[@]}" --output "/tmp/imgread-borrow-replay-results/native-$mode.json"
done
```

The raw result `command` fields retain the exact original run commands/order;
this loop uses the same per-run protocol but interleaves the cases differently
across fresh processes. Default full settings are CPU3, 10 blocks, 2 passes.
To repeat the focused Python validation after building:

```bash
PYTHONPATH="$STATIC/candidate" "$BENCH_PY" -m pytest -q \
  python_tests/test_numpy_output.py python_tests/test_loader_buffers.py \
  python_tests/test_simple_api.py python_tests/test_validation_args.py \
  python_tests/test_resource_limits.py python_tests/test_errors.py \
  python_tests/test_beta_contract.py --basetemp=/tmp/imgread-borrow-replay-tests
PYTHONPATH="$SHARED/candidate" "$BENCH_PY" -m pytest -q \
  python_tests/test_numpy_output.py python_tests/test_loader_buffers.py \
  python_tests/test_simple_api.py python_tests/test_resource_limits.py \
  --basetemp=/tmp/imgread-borrow-replay-shared-tests
```

Rust checks use the same compiler settings, a temporary target and a discoverable
libpython for the embedded-Python unit tests:

```bash
export PATH=/home/albu/Prj/AYA/imgread/.venv/bin:/home/albu/Prj/AYA/imgread/target/loader-toolchain/nasm/usr/bin:$PATH
export CARGO_TARGET_DIR="$STATIC/target" CARGO_BUILD_JOBS=4
export PYO3_PYTHON="$BENCH_PY" CC=/usr/bin/gcc CFLAGS='-O3 -fno-lto'
export TURBOJPEG_SOURCE=vendor TURBOJPEG_STATIC=1 TURBOJPEG_DYNAMIC=0
export LD_LIBRARY_PATH=/home/albu/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib
cargo test --release --locked --offline --features turbojpeg
cargo test --release --locked --offline
cargo clippy --release --locked --offline --all-targets --all-features -- -D warnings
cargo fmt -- --check
ruff check experiments/borrowed_bytes/*.py python_tests/test_numpy_output.py
git diff --check
```
