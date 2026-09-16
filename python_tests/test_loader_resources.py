"""Exact per-instance resource observations, enabled only in diagnostic wheels."""
import io
import os
import warnings

import numpy as np
import pytest
from PIL import Image

import imgread
from test_beta_contract import encode
from test_loader import outcome
from test_resource_limits import oversized


@pytest.fixture(autouse=True)
def diagnostics():
    if not hasattr(imgread.Loader, "_debug_state"):
        if os.environ.get("IMGREAD_REQUIRE_DIAGNOSTICS") == "1":
            pytest.fail("loader-diagnostics feature required")
        pytest.skip("diagnostic wheel only")


def check(loader, cap):
    state = loader._debug_state()
    assert state["pid"] == os.getpid()
    assert state["input_len"] == 0 and state["input_capacity"] <= cap
    assert state["native_live"] <= 1
    return state


@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("cap", [0, 1048576])
def test_warmed_reuse_and_large_error_cleanup(tmp_path, progressive, cap):
    small = tmp_path / "small.jpg"
    small.write_bytes(encode(fmt="JPEG", progressive=progressive))
    large = tmp_path / "large.jpg"
    Image.fromarray(np.random.default_rng(37).integers(0, 256, (1600, 1600, 3), dtype=np.uint8)).save(large, quality=98)
    assert large.stat().st_size > 1048576
    oversized_path = tmp_path / "limit.jpg"
    oversized_path.write_bytes(oversized("JPEG"))
    loader = imgread.Loader([small], max_buffer_bytes=cap)
    initial = check(loader, cap)
    assert initial["input_capacity"] == initial["native_creations"] == 0
    assert initial["manifest_entries"] == 1 and initial["manifest_bytes"] > len(str(small))
    loader[0]
    warm = check(loader, cap)
    for _ in range(10):
        loader[0]
        state = check(loader, cap)
        if cap:
            assert state["input_growths"] == warm["input_growths"]
            assert state["input_capacity"] == warm["input_capacity"]
        assert state["native_creations"] == warm["native_creations"]
    if "turbojpeg" in imgread.supported_backends():
        assert warm["native_creations"] == warm["native_live"] == 1
    loader(large)
    state = check(loader, cap)
    assert state["input_capacity"] == warm["input_capacity"]
    with pytest.raises(ValueError):
        loader(oversized_path)
    assert check(loader, cap)["native_live"] == 0
    with pytest.raises(FileNotFoundError):
        loader(tmp_path / "missing.jpg")
    check(loader, cap)
    loader[0]
    check(loader, cap)


def without_segments(data, marker, remove_all=True):
    output = bytearray(data[:2])
    position = 2
    removed = False
    while position < len(data):
        assert data[position] == 255
        code = data[position + 1]
        if code == 0xDA:
            output.extend(data[position:])
            break
        length = int.from_bytes(data[position + 2:position + 4], "big") + 2
        if code == marker and (remove_all or not removed):
            removed = True
        else:
            output.extend(data[position:position + length])
        position += length
    assert removed
    return bytes(output)


def test_previous_tables_cannot_change_next_result(tmp_path):
    good = tmp_path / "good.jpg"
    suspect = tmp_path / "suspect.jpg"
    good.write_bytes(encode(fmt="JPEG"))
    loader = imgread.Loader()
    for options in ({}, {"optimize": True}, {"progressive": True}):
        full = encode(fmt="JPEG", **options)
        variants = [without_segments(full, marker, all_) for marker in (0xDB, 0xC4) for all_ in (False, True)]
        variants += [full[:full.index(b"\xff\xda")] + b"\xff\xd9"]
        for data in variants:
            loader(good)
            suspect.write_bytes(data)
            assert outcome(loader, suspect) == outcome(imgread.load_numpy, suspect)
            assert check(loader, 1048576)["native_live"] == 0
            assert outcome(loader, good) == outcome(imgread.load_numpy, good)


def test_manifest_and_input_are_not_outputs(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(encode())
    loader = imgread.Loader([path] * 16, backend="image")
    outputs = [loader[0] for _ in range(8)]
    state = check(loader, 1048576)
    assert state["manifest_entries"] == 16
    assert state["native_live"] == state["native_creations"] == 0
    assert sum(array.nbytes for array in outputs) == 8 * 13 * 7 * 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warning_loader = imgread.Loader(backend="turbojpeg", max_buffer_bytes=0)
        with pytest.raises(RuntimeWarning):
            warning_loader(path)
    check(warning_loader, 0)
