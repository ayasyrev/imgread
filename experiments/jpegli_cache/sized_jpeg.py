"""JPEG-only, size-aware helpers for the repeat benchmark."""

from io import BytesIO
from pathlib import Path

import ajpegli
import numpy as np
from experiments.ram_image_codecs.codecs import jpeg_properties
from experiments.ram_image_codecs.contracts import atomic_json, read_json
from experiments.ram_image_codecs.validation import psnr
from experiments.ram_image_codecs.validation import rgb_control as original_rgb_control
from PIL import Image, ImageOps


def prepare_sources(reference, side, output, count):
    manifest = read_json(reference / "manifest.json")
    root = Path(manifest["root"]).resolve(strict=True)
    if count > len(manifest["entries"]):
        raise ValueError("not enough source images in the reference manifest")
    pixels = np.empty((count, side, side, 3), dtype=np.uint8)
    entries = []
    for i, entry in enumerate(manifest["entries"][:count]):
        path = (root / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("source path escapes the original corpus root")
        with Image.open(path) as decoded:
            rgb = ImageOps.exif_transpose(decoded).convert("RGB")
            w, h = rgb.size
            scale = side / min(w, h)
            nw, nh = max(side, round(w * scale)), max(side, round(h * scale))
            rgb = rgb.resize((nw, nh), Image.Resampling.BILINEAR, reducing_gap=None)
            x, y = (nw - side) // 2, (nh - side) // 2
            pixels[i] = np.asarray(rgb.crop((x, y, x + side, y + side)))
        entries.append({"ordinal": i, "path": entry["path"], "source_shape": [h, w, 3]})
    with (output / "prepared.npy").open("xb") as stream:
        np.save(stream, pixels, allow_pickle=False)
    preparation = {
        "reference_manifest": str((reference / "manifest.json").resolve()),
        "root": str(root),
        "seed": manifest["seed"],
        "entries": entries,
        "shape": list(pixels.shape),
        "dtype": "uint8",
        "layout": "C",
        "preparation": f"Original sources; Pillow EXIF transpose; RGB without ICC; bilinear short side {side}, Python round; floor center crop; reducing_gap=None",
        "upscaled_sources": sum(min(e["source_shape"][:2]) < side for e in entries),
        "downscaled_sources": sum(min(e["source_shape"][:2]) > side for e in entries),
        "unchanged_scale_sources": sum(
            min(e["source_shape"][:2]) == side for e in entries
        ),
    }
    atomic_json(output / "manifest.json", preparation)
    return pixels, preparation


def check_array(array, shape):
    if (
        not isinstance(array, np.ndarray)
        or array.shape != shape
        or array.dtype != np.uint8
        or not array.flags.c_contiguous
    ):
        raise ValueError(f"expected C-contiguous RGB uint8 {shape}")


def encode_one(spec, rgb):
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("expected an HWC RGB array")
    check_array(rgb, rgb.shape)
    if spec.family == "jpeg-turbo":
        from PIL import features

        if not features.check_feature("libjpeg_turbo"):
            raise RuntimeError("Pillow is not linked to libjpeg-turbo")
        stream = BytesIO()
        Image.fromarray(rgb).save(stream, format="JPEG", **spec.options)
        data = stream.getvalue()
    elif spec.family == "jpegli":
        data = ajpegli.encode(rgb, **spec.options)
    else:
        raise ValueError(f"unsupported encoder family: {spec.family}")
    if type(data) is not bytes:
        raise TypeError("encoder did not return immutable bytes")
    prop = jpeg_properties(data)
    sampling = (
        [0x22, 0x11, 0x11] if spec.options["subsampling"] in (2, "420") else [0x11] * 3
    )
    if (
        prop["sof"] != (0xC2 if spec.options["progressive"] else 0xC0)
        or prop["precision"] != 8
        or prop["components"] != 3
        or (prop["height"], prop["width"]) != rgb.shape[:2]
        or prop["sampling"] != sampling
        or not prop["dht"]
        or prop["arithmetic"]
    ):
        raise ValueError(f"JPEG stream does not match encoder settings: {prop}")
    return data


def error_metrics(actual, expected):
    check_array(expected, expected.shape)
    check_array(actual, expected.shape)
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    mae = float(np.abs(delta).mean())
    mse = float(np.square(delta).mean())
    return {"mae": mae, "mse": mse, "psnr": psnr(mse)}


def rgb_control(side):
    if side % 256:
        raise ValueError("RGB control requires a multiple of 256")
    scale = side // 256
    return original_rgb_control().repeat(scale, axis=0).repeat(scale, axis=1)


def check_rgb_control(actual, expected):
    check_array(actual, expected.shape)
    scale = expected.shape[0] // 256
    for source_y, source_x in [(40, 40), (100, 180), (208, 40), (208, 180), (48, 192)]:
        y, x, radius = source_y * scale, source_x * scale, 4 * scale
        observed = actual[y - radius : y + radius, x - radius : x + radius].mean(
            axis=(0, 1)
        )
        target = expected[y, x].astype(float)
        if int(np.argmax(observed)) != int(np.argmax(target)):
            raise ValueError("RGB control failed: channels, layout or constant output")
