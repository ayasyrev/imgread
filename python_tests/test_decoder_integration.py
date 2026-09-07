from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

import imgread


def encoded_image_bytes(*, fmt: str = "PNG", color=(10, 20, 30)) -> bytes:
    out = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(out, format=fmt)
    return out.getvalue()


def test_valid_supported_path_decodes_without_error(tmp_path: Path):
    path = tmp_path / "valid.png"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path)

    arr = imgread.load_numpy(str(path))

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2, 3)
    assert arr.dtype == np.uint8
    assert arr[0, 0].tolist() == [10, 20, 30]


def test_load_numpy_from_bytes_matches_path_output(tmp_path: Path):
    path = tmp_path / "valid.png"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path)

    from_path = imgread.load_numpy(str(path))
    from_bytes = imgread.load_numpy_from_bytes(path.read_bytes())

    np.testing.assert_array_equal(from_bytes, from_path)
    assert from_bytes.dtype == np.uint8


def test_load_numpy_from_bytes_accepts_bytearray_and_memoryview():
    data = encoded_image_bytes(color=(12, 34, 56))

    from_bytearray = imgread.load_numpy_from_bytes(bytearray(data))
    from_memoryview = imgread.load_numpy_from_bytes(memoryview(data))

    assert from_bytearray[0, 0].tolist() == [12, 34, 56]
    np.testing.assert_array_equal(from_memoryview, from_bytearray)


def test_load_numpy_from_bytes_accepts_numpy_uint8_buffers():
    data = encoded_image_bytes(color=(3, 4, 5))
    contiguous = np.frombuffer(data, dtype=np.uint8)
    base = np.zeros(len(data) * 2, dtype=np.uint8)
    base[::2] = contiguous
    non_contiguous = base[::2]

    from_contiguous = imgread.load_numpy_from_bytes(contiguous)
    from_non_contiguous = imgread.load_numpy_from_bytes(non_contiguous)

    assert from_contiguous[0, 0].tolist() == [3, 4, 5]
    np.testing.assert_array_equal(from_non_contiguous, from_contiguous)


def test_bgr_output_swaps_red_blue_channels(tmp_path: Path):
    path = tmp_path / "bgr.png"
    Image.new("RGB", (1, 1), color=(11, 22, 33)).save(path)

    arr = imgread.load_numpy(str(path), color="bgr")

    assert arr[0, 0].tolist() == [33, 22, 11]


def test_load_numpy_from_bytes_preserves_bgr_and_case_insensitive_args():
    data = encoded_image_bytes(color=(11, 22, 33))

    arr = imgread.load_numpy_from_bytes(data, color="BGR", dtype="Uint8", backend="IMAGE")

    assert arr[0, 0].tolist() == [33, 22, 11]


def test_image_backend_explicit_mode(tmp_path: Path):
    path = tmp_path / "explicit_image_backend.jpg"
    Image.new("RGB", (2, 2), color=(7, 8, 9)).save(path)

    arr = imgread.load_numpy(str(path), backend="image")

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2, 3)


def test_supported_backends_reports_expected_values():
    backends = imgread.supported_backends()

    assert isinstance(backends, tuple)
    assert "auto" in backends
    assert "image" in backends


def test_turbojpeg_backend_falls_back_with_runtime_warning_for_png(tmp_path: Path):
    path = tmp_path / "for_turbo_backend.png"
    Image.new("RGB", (2, 2), color=(4, 5, 6)).save(path)

    with pytest.warns(RuntimeWarning) as caught:
        arr = imgread.load_numpy(str(path), backend="turbojpeg")
        arr2 = imgread.load_numpy(str(path), backend="turbojpeg")

    assert len(caught) == 2
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2, 3)
    assert isinstance(arr2, np.ndarray)
