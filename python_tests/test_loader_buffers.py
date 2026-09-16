"""Buffer ownership, error recovery and parity for the persistent decoder."""
import gc
import inspect
import sys
import warnings
import weakref

import numpy as np
import pytest

import imgread
from test_beta_contract import encode
from test_loader import CASES, outcome
from test_resource_limits import oversized


@pytest.mark.parametrize("mode,fmt,options", CASES)
@pytest.mark.parametrize("backend", ["auto", "image", "turbojpeg"])
@pytest.mark.parametrize("color", ["rgb", "bgr"])
@pytest.mark.parametrize("limits", ["safe", "unlimited"])
def test_buffer_function_parity(mode, fmt, options, backend, color, limits):
    data = encode(mode, fmt, **options)
    loader = imgread.Loader(backend=backend, color=color, limits=limits)
    expected = outcome(lambda data: imgread.load_numpy_from_bytes(
        data, backend=backend, color=color, limits=limits), data)
    for _ in range(2):
        assert outcome(loader.decode, data) == expected


def buffers(data):
    yield data
    yield bytearray(data)
    yield memoryview(data)
    yield memoryview(bytearray(data)).toreadonly()
    yield np.frombuffer(data, dtype="u1")
    padded = np.zeros(len(data) * 2, dtype="u1")
    padded[::2] = np.frombuffer(data, dtype="u1")
    yield padded[::2]
    yield memoryview(padded[::2])
    yield np.frombuffer(data[::-1], dtype="u1")[::-1]
    yield np.frombuffer(data, dtype="u1").reshape(1, -1)


def test_buffer_types_and_output_ownership():
    data = encode(fmt="JPEG")
    loader = imgread.Loader()
    assert str(inspect.signature(loader.decode)) == "(data)"
    expected = imgread.load_numpy_from_bytes(data)
    outputs = [loader.decode(value) for value in buffers(data)]
    mutable = bytearray(data)
    outputs.append(loader.decode(mutable))
    mutable[:] = bytes(len(mutable))
    # Neither an immutable argument nor a buffer exporter is retained.
    refs = sys.getrefcount(data)
    loader.decode(data)
    assert sys.getrefcount(data) == refs
    value = np.frombuffer(data, dtype="u1").copy()
    reference = weakref.ref(value)
    outputs.append(loader.decode(value))
    del value, loader
    gc.collect()
    assert reference() is None
    for actual in outputs:
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.writeable and actual.flags.c_contiguous
    outputs[0][:] = 0
    for actual in outputs[1:]:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad", [None, "image.jpg", 1, [1, 2], np.zeros(3, dtype="u2")])
def test_invalid_buffer_and_recovery(bad):
    loader = imgread.Loader()
    with pytest.raises(TypeError, match="uint8-compatible buffer"):
        loader.decode(bad)
    assert loader.decode(encode()).shape == (7, 13, 3)


@pytest.mark.parametrize("backend", ["auto", "image", "turbojpeg"])
def test_errors_warnings_limits_and_recovery(backend):
    loader = imgread.Loader(backend=backend)
    function = lambda data: imgread.load_numpy_from_bytes(data, backend=backend)
    for data in (encode(fmt="JPEG"), b"", b"\xff\xd8\xff\x00", oversized("JPEG"),
                 encode(), encode("CMYK", "JPEG"), encode(fmt="JPEG")):
        assert outcome(loader.decode, data) == outcome(function, data)
    huge = np.broadcast_to(np.zeros(1, dtype="u1"), (256 * 1024 * 1024 + 1,))
    with pytest.raises(ValueError, match="max_input_bytes"):
        loader.decode(huge)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warning_loader = imgread.Loader(backend="turbojpeg")
        with pytest.raises(RuntimeWarning):
            warning_loader.decode(encode())
    assert outcome(warning_loader.decode, encode(fmt="JPEG")) == outcome(
        lambda data: imgread.load_numpy_from_bytes(data, backend="turbojpeg"), encode(fmt="JPEG"))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Python buffer protocol needs 3.12")
def test_buffer_protocol_callbacks_cannot_reenter_loader():
    loader = imgread.Loader()
    data = encode()

    class Exporter:
        def __buffer__(self, flags):
            with pytest.raises(RuntimeError, match="Loader is busy"):
                loader.decode(data)
            return memoryview(data)

        def __release_buffer__(self, view):
            with pytest.raises(RuntimeError, match="Loader is busy"):
                loader.decode(data)

    np.testing.assert_array_equal(loader.decode(Exporter()), loader.decode(data))


@pytest.mark.parametrize("cap", [0, 1048576])
def test_buffer_reuses_native_state_without_retaining_input(cap, tmp_path):
    loader = imgread.Loader(max_buffer_bytes=cap)
    if not hasattr(loader, "_debug_state"):
        pytest.skip("diagnostic wheel only")
    data = encode(fmt="JPEG")
    for value in buffers(data):
        loader.decode(value)
        state = loader._debug_state()
        assert state["input_capacity"] == state["input_growths"] == 0
        if "turbojpeg" in imgread.supported_backends():
            assert state["native_creations"] == state["native_live"] == 1
    path = tmp_path / "image.jpg"
    path.write_bytes(data)
    loader(path)
    before = loader._debug_state()
    loader.decode(data)
    assert loader._debug_state() == before


def test_previous_jpeg_tables_cannot_affect_buffer_decode():
    from test_loader_resources import without_segments
    loader = imgread.Loader()
    for options in ({}, {"optimize": True}, {"progressive": True}):
        data = encode(fmt="JPEG", **options)
        for marker in (0xDB, 0xC4):
            loader.decode(data)
            broken = without_segments(data, marker)
            assert outcome(loader.decode, broken) == outcome(imgread.load_numpy_from_bytes, broken)
            assert outcome(loader.decode, data) == outcome(imgread.load_numpy_from_bytes, data)
