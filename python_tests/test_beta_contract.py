import errno
import inspect
import io
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

import imgread as lib


def encode(mode="RGB", fmt="PNG", **options):
    values = {"RGB": (10, 80, 160), "RGBA": (10, 80, 160, 25), "L": 83, "CMYK": (5, 35, 80, 10), "I;16": 32768}
    out = io.BytesIO()
    Image.new(mode, (13, 7), values[mode]).save(out, format=fmt, **options)
    return out.getvalue()


@pytest.mark.parametrize("mode,fmt,options", [
    ("RGB", "JPEG", {}), ("RGB", "JPEG", {"progressive": True}), ("CMYK", "JPEG", {}),
    ("L", "PNG", {}), ("RGBA", "PNG", {}), ("I;16", "PNG", {}),
    ("RGB", "TIFF", {"compression": "raw"}), ("RGB", "TIFF", {"compression": "tiff_lzw"}),
    ("RGB", "TIFF", {"compression": "tiff_adobe_deflate"}), ("I;16", "TIFF", {}),
])
def test_corpus(mode, fmt, options, tmp_path):
    data = encode(mode, fmt, **options)
    path = tmp_path / "fixture"
    path.write_bytes(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for backend in ("image", "auto", "turbojpeg"):
            rgb = lib.load_numpy(path, backend=backend)
            np.testing.assert_array_equal(rgb, lib.load_numpy_from_bytes(data, backend=backend))
            np.testing.assert_array_equal(rgb[..., ::-1], lib.load_numpy_from_bytes(data, backend=backend, color="bgr"))
            assert rgb.dtype == np.uint8 and rgb.shape == (7, 13, 3)
            assert rgb.flags.c_contiguous and rgb.flags.writeable
            # Reference semantics: alpha is dropped; uint16 is rounded to 8 bits.
            if mode == "I;16":
                assert np.all(rgb == 128)
            else:
                expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
                assert np.abs(rgb.astype(int) - expected.astype(int)).max() <= (3 if fmt == "JPEG" else 0)


def test_jpeg_backend_parity():
    # A textured 4:4:4 fixture avoids implementation-specific chroma upsampling.
    pixels = np.random.default_rng(4).integers(0, 256, (47, 63, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="JPEG", subsampling=0, quality=93)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = lib.load_numpy_from_bytes(out.getvalue(), backend="image")
        b = lib.load_numpy_from_bytes(out.getvalue(), backend="turbojpeg")
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 3


def test_tiff_endian_and_first_page():
    for mode in ("I;16", "I;16B"):
        out = io.BytesIO()
        Image.new(mode, (4, 3), 32768).save(out, format="TIFF")
        assert np.all(lib.load_numpy_from_bytes(out.getvalue()) == 128)
    out = io.BytesIO()
    Image.new("RGB", (3, 2), (1, 2, 3)).save(out, format="TIFF", save_all=True, append_images=[Image.new("RGB", (8, 9), (9, 8, 7))])
    result = lib.load_numpy_from_bytes(out.getvalue())
    assert result.shape == (2, 3, 3)
    assert result[0, 0].tolist() == [1, 2, 3]


@pytest.mark.parametrize("function", [lib.load_numpy, lib.load_numpy_simple])
def test_pathlike_and_io(function, tmp_path):
    class CustomPath:
        def __fspath__(self):
            return str(tmp_path / "image.jpg")
    (tmp_path / "image.jpg").write_bytes(encode(fmt="JPEG"))
    assert function(CustomPath()).shape == (7, 13, 3)
    for bad in (b"image.jpg", 12, None):
        with pytest.raises(TypeError):
            function(bad)
    with pytest.raises(FileNotFoundError) as caught:
        function(tmp_path / "missing")
    assert caught.value.errno == errno.ENOENT
    with pytest.raises(IsADirectoryError) as caught:
        function(tmp_path)
    assert caught.value.errno == errno.EISDIR
    with pytest.raises(NotADirectoryError) as caught:
        function(tmp_path / "image.jpg" / "child")
    assert caught.value.errno == errno.ENOTDIR
    private = tmp_path / "private.jpg"
    private.write_bytes(encode(fmt="JPEG"))
    private.chmod(0)
    try:
        if os.geteuid() != 0:
            with pytest.raises(PermissionError) as caught:
                function(private)
            assert caught.value.errno == errno.EACCES
    finally:
        private.chmod(0o600)


@pytest.mark.parametrize("policy,count", [("always", 3), ("default", 1), ("ignore", 0)])
def test_warning_filters(policy, count):
    data = encode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        for _ in range(3):
            lib.load_numpy_from_bytes(data, backend="turbojpeg")
    assert len(caught) == count
    for warning in caught:
        assert warning.category is RuntimeWarning
        assert warning.filename == __file__


def test_warning_error_and_stacklevel():
    data = encode()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            with pytest.raises(RuntimeWarning):
                lib.load_numpy_from_bytes(data, backend="turbojpeg")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        lib.load_numpy_from_bytes(data, backend="turbojpeg")
    assert caught[0].lineno == line


def test_simple_fallback_and_fast_path():
    data = encode(fmt="JPEG")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            assert lib.load_numpy_simple_from_bytes(data).shape == (7, 13, 3)
    assert len(caught) == (0 if "turbojpeg" in lib.supported_backends() else 2)
    # TurboJPEG cannot convert CMYK directly to RGB, but image can.
    with pytest.warns(RuntimeWarning):
        lib.load_numpy_simple_from_bytes(encode("CMYK", "JPEG"))


@pytest.mark.parametrize("function", [lib.load_numpy_from_bytes, lib.load_numpy_simple_from_bytes])
def test_strided_buffers_and_limits_arguments(function):
    data = encode(fmt="JPEG")
    strided = np.zeros(len(data) * 2, dtype="u1")
    strided[::2] = np.frombuffer(data, dtype="u1")
    np.testing.assert_array_equal(function(strided[::2]), function(data, limits="unlimited"))
    reverse = np.frombuffer(data[::-1], dtype="u1")[::-1]
    np.testing.assert_array_equal(function(reverse), function(data))
    with pytest.raises(ValueError, match="limits"):
        function(data, limits="fast")
    with pytest.raises(TypeError):
        function(np.zeros(3, dtype="u2"))


def test_concurrent_decodes():
    data = encode(fmt="JPEG", progressive=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lib.load_numpy_from_bytes, [data] * 32))
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "TIFF"])
def test_truncated_payload(fmt):
    data = encode(fmt=fmt)
    with pytest.raises((ValueError, RuntimeError)):
        lib.load_numpy_from_bytes(data[:12])
