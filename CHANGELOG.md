# Changelog

## 0.2.1 — 2026-09-16

- Add `imgread.Loader` for repeated decoding with fixed options:
  `loader(path)`, `Loader(paths)[index]`, and `loader.decode(data)`.
- Reuse bounded file-input storage and native JPEG decoder state within each
  Loader. Every result remains an independent writable NumPy array.
- Borrow immutable `bytes` during `Loader.decode`; copy other input buffers
  before releasing the GIL. Input buffers are not retained after the call.
- Support pickling and PyTorch DataLoader workers, with decoder state initialized
  separately in each process. Preserve filesystem path spelling through pickle.
- Reject overlapping calls on one Loader and preserve resource-limit, fallback,
  and corrupt-input recovery behavior.

Existing functional decoding APIs remain available. Loader reuse does not
guarantee a throughput improvement; measure it in the consuming pipeline.

## 0.2.0 — 2026-09-07

- Publish bounded JPEG, PNG, and TIFF decoding into RGB/BGR NumPy arrays, with
  path and buffer inputs, JPEG-only convenience functions, and a statically
  linked TurboJPEG backend in the official wheels.
