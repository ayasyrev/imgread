import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import warnings
import zlib

import numpy as np
import pytest
from PIL import Image

import imgread as lib


def oversized(fmt, width=65000, height=65000):
    out = io.BytesIO()
    Image.new("RGB", (2, 2)).save(out, format=fmt)
    data = bytearray(out.getvalue())
    if fmt == "JPEG":
        start = data.index(b"\xff\xc0")
        data[start + 5:start + 9] = struct.pack(">HH", height, width)
    elif fmt == "PNG":
        data[16:24] = struct.pack(">II", width, height)
        data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    else:
        order = "<" if data[:2] == b"II" else ">"
        ifd = struct.unpack_from(order + "I", data, 4)[0]
        count = struct.unpack_from(order + "H", data, ifd)[0]
        for offset in range(ifd + 2, ifd + 2 + count * 12, 12):
            tag, kind = struct.unpack_from(order + "HH", data, offset)
            if tag in (256, 257, 278):
                value = width if tag == 256 else height
                struct.pack_into(order + ("H" if kind == 3 else "I"), data, offset + 8, value)
    return bytes(data)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "TIFF"])
@pytest.mark.parametrize("backend", ["image", "turbojpeg", "auto"])
@pytest.mark.parametrize("source", ["path", "bytes"])
def test_oversized_all_routes(fmt, backend, source, tmp_path):
    data = oversized(fmt)
    path = tmp_path / "oversized"
    path.write_bytes(data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            if source == "path":
                lib.load_numpy(path, backend=backend)
            else:
                lib.load_numpy_from_bytes(data, backend=backend)
        if fmt == "JPEG":
            with pytest.raises(ValueError):
                if source == "path":
                    lib.load_numpy_simple(path)
                else:
                    lib.load_numpy_simple_from_bytes(data)
    assert not caught


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "TIFF"])
def test_rejected_before_large_allocation(fmt, tmp_path):
    path = tmp_path / "oversized"
    path.write_bytes(oversized(fmt))
    code = '''
import json, resource, sys, warnings
import imgread as lib
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with warnings.catch_warnings():
    warnings.simplefilter("error")
    for backend in ("auto", "image", "turbojpeg"):
        try: lib.load_numpy(sys.argv[1], backend=backend)
        except ValueError: pass
        else: raise AssertionError("oversized image accepted")
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"rss_delta_bytes": (after-before) * (1 if sys.platform == "darwin" else 1024)}))
'''
    result = subprocess.run([sys.executable, "-c", code, str(path)], check=True, capture_output=True, text=True, timeout=20)
    assert json.loads(result.stdout)["rss_delta_bytes"] < 32 * 1024 * 1024


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux /proc and RLIMIT_AS safety guard")
@pytest.mark.parametrize("source", ["path", "bytes"])
def test_progressive_jpeg_working_set_rejected_before_allocation(source, tmp_path):
    out = io.BytesIO()
    Image.new("RGB", (2, 2)).save(out, format="JPEG", progressive=True, subsampling=0)
    data = bytearray(out.getvalue())
    start = data.index(b"\xff\xc2")
    # Final RGB pixels fit SAFE, but progressive 4:4:4 coefficient planes do not.
    data[start + 5:start + 9] = struct.pack(">HH", 10000, 10000)
    path = tmp_path / "progressive.jpg"
    path.write_bytes(data)
    code = '''
import json, os, resource, sys, warnings
from pathlib import Path
import imgread as lib

data = Path(sys.argv[1]).read_bytes()
# Bound the child after importing NumPy/the extension: a missing preflight must
# fail this subprocess safely, not exhaust memory on the test runner.
with open("/proc/self/statm") as statm:
    virtual_bytes = int(statm.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
_, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
limit = virtual_bytes + 128 * 1024 * 1024
if hard_limit != resource.RLIM_INFINITY:
    limit = min(limit, hard_limit)
resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
        if sys.argv[2] == "path":
            lib.load_numpy(sys.argv[1], backend="image")
        else:
            lib.load_numpy_from_bytes(data, backend="image")
    except ValueError as error:
        assert "max_decoder_alloc" in str(error), str(error)
    else:
        raise AssertionError("progressive JPEG working set accepted")
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"rss_delta_bytes": (after - before) * 1024}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), source],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rss_delta_bytes"] < 32 * 1024 * 1024


@pytest.mark.parametrize("mode,subsampling", [("RGB", 0), ("RGB", 1), ("RGB", 2), ("L", 0), ("CMYK", 0)])
@pytest.mark.parametrize("limits", ["safe", "unlimited"])
def test_small_progressive_jpeg_with_limits(mode, subsampling, limits, tmp_path):
    colors = {"RGB": (10, 80, 160), "L": 83, "CMYK": (5, 35, 80, 10)}
    out = io.BytesIO()
    Image.new(mode, (17, 9), colors[mode]).save(
        out, format="JPEG", progressive=True, subsampling=subsampling,
    )
    data = out.getvalue()
    path = tmp_path / "progressive.jpg"
    path.write_bytes(data)
    expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pixels = lib.load_numpy_from_bytes(data, backend="image", limits=limits)
        np.testing.assert_array_equal(pixels, lib.load_numpy(path, backend="image", limits=limits))
    assert pixels.shape == (9, 17, 3)
    assert pixels.dtype == np.uint8
    assert np.abs(pixels.astype(int) - expected.astype(int)).max() <= 3


@pytest.mark.parametrize("simple", [False, True])
def test_input_cap_before_buffer_copy_and_file_read(simple, tmp_path):
    # A zero-stride view exposes 256 MiB + 1 logically without allocating it.
    data = np.broadcast_to(np.zeros(1, dtype="u1"), (256 * 1024 * 1024 + 1,))
    function = lib.load_numpy_simple_from_bytes if simple else lib.load_numpy_from_bytes
    with pytest.raises(ValueError, match="max_input_bytes"):
        function(data)
    path = tmp_path / "sparse.jpg"
    with path.open("wb") as file:
        file.truncate(256 * 1024 * 1024 + 1)
    function = lib.load_numpy_simple if simple else lib.load_numpy
    with pytest.raises(ValueError, match="max_input_bytes"):
        function(path)


def test_dimension_boundary_and_unlimited():
    out = io.BytesIO()
    Image.new("RGB", (32768, 1), (1, 2, 3)).save(out, format="PNG")
    assert lib.load_numpy_from_bytes(out.getvalue()).shape == (1, 32768, 3)
    out = io.BytesIO()
    Image.new("RGB", (32769, 1), (1, 2, 3)).save(out, format="PNG")
    with pytest.raises(ValueError):
        lib.load_numpy_from_bytes(out.getvalue())
    assert lib.load_numpy_from_bytes(out.getvalue(), limits="unlimited").shape == (1, 32769, 3)


@pytest.mark.parametrize("source", ["path", "bytes"])
@pytest.mark.parametrize("backend", ["image", "auto", "turbojpeg"])
def test_codec_dimension_capability_error_falls_back(source, backend, tmp_path):
    data = oversized("JPEG", width=65535, height=1)
    path = tmp_path / "maximum-dimension.jpg"
    path.write_bytes(data)
    turbojpeg_enabled = "turbojpeg" in lib.supported_backends()
    expects_warning = backend == "turbojpeg" or (backend == "auto" and turbojpeg_enabled)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if source == "path":
            pixels = lib.load_numpy(path, backend=backend, limits="unlimited")
        else:
            pixels = lib.load_numpy_from_bytes(data, backend=backend, limits="unlimited")

    assert pixels.shape == (1, 65535, 3)
    assert len(caught) == int(expects_warning)
    if turbojpeg_enabled and backend != "image":
        assert "Maximum supported image dimension" in str(caught[0].message)
