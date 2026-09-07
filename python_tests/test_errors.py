import os
from pathlib import Path

import pytest

import imgread


def test_missing_file_raises_file_not_found_error(tmp_path: Path):
    missing = tmp_path / "does-not-exist.jpg"

    with pytest.raises(FileNotFoundError):
        imgread.load_numpy(str(missing))


@pytest.mark.skipif(os.name != "posix", reason="POSIX filenames are raw bytes")
@pytest.mark.parametrize("function", [imgread.load_numpy, imgread.load_numpy_simple])
def test_missing_file_preserves_non_utf8_filename(function, tmp_path: Path):
    missing_bytes = os.fsencode(tmp_path) + b"/does-not-exist-\xff.jpg"
    missing = os.fsdecode(missing_bytes)

    with pytest.raises(FileNotFoundError) as caught:
        function(missing)

    assert caught.value.filename == missing
    assert os.fsencode(caught.value.filename) == missing_bytes


def test_unsupported_format_raises_value_error(tmp_path: Path):
    path = tmp_path / "not-an-image.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        imgread.load_numpy(str(path))


def test_corrupt_image_raises_runtime_error(tmp_path: Path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not a real jpeg payload")

    with pytest.raises(RuntimeError):
        imgread.load_numpy(str(path))


def test_unsupported_bytes_raise_value_error():
    with pytest.raises(ValueError):
        imgread.load_numpy_from_bytes(b"hello")


def test_corrupt_jpeg_bytes_raise_runtime_error():
    with pytest.raises(RuntimeError):
        imgread.load_numpy_from_bytes(b"\xFF\xD8\xFF not a real jpeg payload")


def test_non_buffer_bytes_input_raises_type_error():
    with pytest.raises(TypeError):
        imgread.load_numpy_from_bytes(123)
