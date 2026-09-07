from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import imgread


def test_color_and_dtype_are_case_insensitive(tmp_path: Path):
    path = tmp_path / "ok.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(path)

    imgread.load_numpy(str(path), color="RGB", dtype="Uint8")


def test_invalid_color_still_raises_value_error(tmp_path: Path):
    path = tmp_path / "ok2.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(path)

    with pytest.raises(ValueError):
        imgread.load_numpy(str(path), color="invalid", dtype="uint8")


def test_invalid_backend_still_raises_value_error(tmp_path: Path):
    path = tmp_path / "ok3.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(path)

    with pytest.raises(ValueError):
        imgread.load_numpy(str(path), backend="unknown-backend")


def test_load_numpy_from_bytes_invalid_backend_raises_value_error():
    with pytest.raises(ValueError):
        imgread.load_numpy_from_bytes(b"unused", backend="unknown-backend")


def test_load_numpy_from_bytes_non_uint8_buffer_raises_type_error():
    data = np.array([1, 2, 3, 4], dtype=np.uint16)

    with pytest.raises(TypeError):
        imgread.load_numpy_from_bytes(data)
