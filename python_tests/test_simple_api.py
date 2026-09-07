from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import imgread


def encoded_jpeg_bytes(*, color=(10, 20, 30)) -> bytes:
    out = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(out, format="JPEG")
    return out.getvalue()


def test_load_numpy_simple_decodes_jpeg_path(tmp_path: Path):
    path = tmp_path / "simple.jpg"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path, format="JPEG")

    arr = imgread.load_numpy_simple(str(path))

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (2, 2, 3)


def test_load_numpy_simple_from_bytes_matches_path(tmp_path: Path):
    path = tmp_path / "simple_bytes.jpg"
    Image.new("RGB", (2, 2), color=(40, 50, 60)).save(path, format="JPEG")

    from_path = imgread.load_numpy_simple(str(path))
    from_bytes = imgread.load_numpy_simple_from_bytes(path.read_bytes())

    np.testing.assert_array_equal(from_bytes, from_path)
    assert from_bytes.dtype == np.uint8
    assert from_bytes.shape == (2, 2, 3)


def test_load_numpy_simple_from_bytes_accepts_buffer_inputs():
    data = encoded_jpeg_bytes(color=(70, 80, 90))

    from_bytes = imgread.load_numpy_simple_from_bytes(data)
    from_bytearray = imgread.load_numpy_simple_from_bytes(bytearray(data))
    from_memoryview = imgread.load_numpy_simple_from_bytes(memoryview(data))
    from_numpy = imgread.load_numpy_simple_from_bytes(np.frombuffer(data, dtype=np.uint8))

    np.testing.assert_array_equal(from_bytearray, from_bytes)
    np.testing.assert_array_equal(from_memoryview, from_bytes)
    np.testing.assert_array_equal(from_numpy, from_bytes)


def test_load_numpy_simple_from_bytes_rejects_non_buffer_input():
    with pytest.raises(TypeError, match="uint8-compatible buffer"):
        imgread.load_numpy_simple_from_bytes(123)


def test_load_numpy_simple_from_bytes_rejects_non_uint8_buffer():
    data = np.array([1, 2, 3, 4], dtype=np.uint16)

    with pytest.raises(TypeError, match="uint8-compatible buffer"):
        imgread.load_numpy_simple_from_bytes(data)


def test_load_numpy_simple_from_bytes_rejects_non_jpeg_bytes():
    with pytest.raises(ValueError):
        imgread.load_numpy_simple_from_bytes(b"not a jpeg")


def test_load_numpy_simple_from_bytes_corrupt_jpeg_raises_runtime_error():
    with pytest.raises(RuntimeError):
        imgread.load_numpy_simple_from_bytes(b"\xFF\xD8\xFF not a real jpeg payload")
