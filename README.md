# imgread

A **public beta** Python library for decoding JPEG, PNG and TIFF into NumPy arrays,
with a Rust decoder and a statically linked libjpeg-turbo backend.

```sh
python -m pip install --pre imgread
```

Use a wheel on a supported platform; it needs no Rust toolchain. To require a wheel:
`python -m pip install --pre --only-binary=:all: imgread`.
A source build requires Rust 1.88 or newer, CMake, a C compiler and NASM.

```python
from pathlib import Path
import imgread

# Path and os.PathLike[str]
rgb = imgread.load_numpy(Path("photo.jpg"))

# bytes, bytearray, memoryview or a uint8 NumPy buffer (including strided views)
bgr = imgread.load_numpy_from_bytes(Path("photo.png").read_bytes(), color="bgr")

# Choose the portable Rust decoder explicitly
rgb = imgread.load_numpy("scan.tiff", backend="image")
```

## Array and format contract

Every successful call returns a new writable, C-contiguous `numpy.ndarray` of
shape `(height, width, 3)` and dtype `uint8`. The default order is RGB; `color="bgr"`
reverses the channels. Only `dtype="uint8"` is supported. Color, dtype, backend and
limit-profile names are case-insensitive. Bytes paths are rejected; use the buffer
APIs for encoded bytes. Decoding releases the GIL after taking a private input copy.

- JPEG: baseline, progressive, grayscale and CMYK. CMYK may use the fallback
  decoder. Decoder rounding and chroma upsampling can produce small pixel
  differences between backends; bit-for-bit JPEG parity is not promised.
- PNG: grayscale, RGB, RGBA and 16-bit input. Alpha is discarded without compositing.
  Grayscale is replicated into three channels. Unsigned 16-bit values are rounded
  to 8 bits using `(value + 128) // 257`.
- TIFF: the **first page only**, without applying orientation. Common uncompressed,
  LZW and Deflate variants and little-/big-endian samples are supported. Supported
  encodings follow the `image` TIFF decoder; unsupported encodings raise errors.
- EXIF orientation, ICC color conversion and other metadata are not applied or
  returned. Pixels remain in the encoded color space. Animated PNG returns its
  default image, not an animation.

Other formats are rejected even though the underlying Rust `image` dependency
retains its default features in this beta.

## Backends and the simple API

`backend="auto"` selects TurboJPEG for JPEG when compiled in and `image` otherwise.
`backend="image"` always selects the Rust decoder. `backend="turbojpeg"` falls back
to `image` when unavailable, incompatible with the format, or unable to decode it.
A successful fallback emits `RuntimeWarning`. An unsuccessful decode raises an error.
`auto` also warns if an attempted TurboJPEG decode falls back. Inspect the build
with `imgread.supported_backends()`; official wheels include TurboJPEG.

`load_numpy_simple(path, *, limits="safe")` and
`load_numpy_simple_from_bytes(data, *, limits="safe")` are JPEG-only RGB shortcuts.
They use the same checked decoder and warning/fallback behavior. Non-JPEG input is
always a `ValueError`. Python's normal `warnings` filters control every warning,
including `always`, `ignore` and `error`; the library has no global deduplication.

## Resource limits

All four decode functions accept the keyword-only argument `limits="safe"`:

| Budget per image | Default cap |
| --- | ---: |
| Encoded input | 256 MiB |
| Width / height | 32,768 each |
| Pixels | 100,000,000 |
| RGB/BGR output | 512 MiB |
| Decoder allocation budget | 512 MiB |

Input size is checked before copying a Python buffer or reading a regular file.
The open file is read with a bound even if it grows. Dimensions and output sizes
are checked before the pixel allocation. A resource-limit failure is terminal:
**no fallback** can retry it with another decoder.

The decoder budget is passed to `image::Limits` and to libjpeg-turbo's intermediate
buffer memory limit. The `image` JPEG backend also preflights a conservative
working-set bound covering padded coefficient planes, decoded pixels and
row/upsampling buffers, because its upstream JPEG decoder ignores `max_alloc`.
This applies to fallback routes too, and may reject a large JPEG even when its
final pixels fit the output cap (including baseline JPEGs that need less memory).
These budgets do not measure total process RSS:
encoded input, output, metadata, conversion buffers, allocator overhead and
concurrent calls can add to it. It is a per-image policy, not a process sandbox.

For trusted large inputs only, use `limits="unlimited"`. This removes policy caps;
checked arithmetic, address-space checks and fallible application allocations
remain enabled. It makes no speed promise. Ordinary allocation pressure and
upstream codec behavior can still cause errors.

## Errors

| Condition | Exception |
| --- | --- |
| Invalid input type, including bytes paths | `TypeError` |
| Invalid option, unsupported format or resource limit | `ValueError` |
| Missing file | `FileNotFoundError` |
| Permission denied | `PermissionError` |
| Directory used as input | `IsADirectoryError` |
| Other filesystem failure | `OSError` subclass with OS `errno` |
| Corrupt or undecodable image | `RuntimeError` |
| Successful backend fallback | `RuntimeWarning` |

Format detection uses content; for path input an extension is a last resort for
identifying a damaged image. Unknown bytes raise `ValueError`; a recognized but
corrupt image raises `RuntimeError`.

## Beta support and stability

The release wheel matrix targets CPython **3.11–3.14**, Linux x86_64
manylinux2014 (glibc 2.17+) and macOS x86_64 (10.13+) / arm64 (11.0+).
Windows, Linux aarch64, musllinux, PyPy, free-threaded CPython and Python 3.15 are
not supported by this beta. The 3.15 compatibility CI job is informational.

The five functions above form the beta API. Bug fixes can change rejection of
malformed inputs, limits or decoder results. Intentional API changes will be
recorded in release notes and beta versions; pin a version for reproducible work.
The Rust crate is internal and is not published to crates.io.

## Development

Use `uv run` for Python-facing commands. Normal builds and tests use the committed
`Cargo.lock` and `uv.lock` without updating them.

```sh
uv sync --locked --no-install-project
uv run maturin develop --locked
uv run pytest python_tests -q
cargo test --locked
cargo test --locked --features turbojpeg
cargo fmt -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
uv run maturin build --release --locked --sdist
uv run twine check target/wheels/*
uv run python scripts/check_artifacts.py --allow-local target/wheels/*
uv run python scripts/update_notices.py --check
```

`--allow-local` validates local artifacts without certifying their platform tags
for release. Keep maturin's configured features when building; passing
`--features extension-module` alone overrides them and removes TurboJPEG.

## License

The project uses the MIT license. Redistributed dependency notices are in
`THIRD_PARTY_NOTICES.md` and `licenses/` and are included in both wheels and sdists.

This software is based in part on the work of the Independent JPEG Group.
