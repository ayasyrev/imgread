"""Native JPEG output is NumPy-owned, including allocator lifetime/error policy."""

import gc
import importlib.util
import shlex
import shutil
import subprocess
import sysconfig
import warnings
from pathlib import Path

import numpy as np
import pytest

import imgread
from test_beta_contract import encode


pytestmark = pytest.mark.skipif(
    "turbojpeg" not in imgread.supported_backends(), reason="native JPEG build required"
)


@pytest.fixture(scope="module")
def memory_policy(tmp_path_factory):
    compiler = shlex.split(sysconfig.get_config_var("LDSHARED"))
    if not shutil.which(compiler[0]):
        pytest.skip("C compiler required for the NumPy allocator probe")
    directory = tmp_path_factory.mktemp("numpy-output-policy")
    target = directory / (
        "_numpy_output_probe" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    subprocess.run(
        compiler
        + [
            "-fPIC",
            "-O2",
            "-I" + sysconfig.get_path("include"),
            "-I" + np.get_include(),
            str(Path(__file__).with_name("numpy_output_probe.c")),
            "-o",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    spec = importlib.util.spec_from_file_location("_numpy_output_probe", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "mode,options", [("RGB", {}), ("RGB", {"progressive": True}), ("L", {})]
)
@pytest.mark.parametrize("backend", ["auto", "turbojpeg"])
@pytest.mark.parametrize("color", ["rgb", "bgr"])
def test_all_jpeg_entrypoints_own_output(mode, options, backend, color, tmp_path):
    data = encode(mode, "JPEG", **options)
    path = tmp_path / "image.jpg"
    path.write_bytes(data)
    loader = imgread.Loader([path], backend=backend, color=color)
    outputs = [
        imgread.load_numpy_from_bytes(data, backend=backend, color=color),
        imgread.load_numpy(path, backend=backend, color=color),
        loader.decode(data),
        loader(path),
        loader[0],
    ]
    if color == "rgb":
        outputs += [
            imgread.load_numpy_simple(path),
            imgread.load_numpy_simple_from_bytes(data),
        ]
    expected = outputs[0].copy()
    del data, loader
    gc.collect()
    for array in outputs:
        assert array.base is None and array.flags.owndata
        assert array.flags.writeable and array.flags.c_contiguous
        np.testing.assert_array_equal(array, expected)
    outputs[0].fill(0)
    for array in outputs[1:]:
        np.testing.assert_array_equal(array, expected)


def test_numpy_allocator_owns_pixels_until_last_view_dies(memory_policy):
    data = encode(fmt="JPEG")
    policy = memory_policy
    old = policy.install(7 * 13 * 3)
    try:
        array = imgread.load_numpy_from_bytes(data)
        view = array[::-1, ::2]
        assert policy.stats() == (1, 0, 0, 1)
    finally:
        policy.restore(old)
    del array
    gc.collect()
    assert policy.stats()[:3] == (1, 0, 0)
    assert view.shape == (7, 7, 3) and view.flags.writeable
    del view
    gc.collect()
    assert policy.stats()[:3] == (1, 1, 0)


@pytest.mark.parametrize("persistent", [False, True])
def test_output_allocation_failure_is_terminal_and_recovers(memory_policy, persistent):
    data = encode(fmt="JPEG")
    decode = (
        imgread.Loader(backend="turbojpeg").decode
        if persistent
        else imgread.load_numpy_simple_from_bytes
    )
    decode(data)  # Warm native reuse and NumPy's API initialization first.
    old = memory_policy.install(7 * 13 * 3, None, True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="output allocation failed"):
                decode(data)
        assert memory_policy.stats() == (0, 0, 0, 1)
    finally:
        memory_policy.restore(old)
    result = decode(data)
    assert result.shape == (7, 13, 3) and result.flags.owndata


@pytest.mark.parametrize("persistent", [False, True])
def test_native_failure_frees_private_array_before_fallback(memory_policy, persistent):
    data = encode(fmt="JPEG")
    scan = data.index(b"\xff\xda")
    entropy = scan + 2 + int.from_bytes(data[scan + 2 : scan + 4], "big")
    # The header is valid; a second SOF in the entropy payload fails only after
    # native output allocation. The image backend tolerates this fixture.
    damaged = data[:entropy] + b"\xff\xc0\x00\x01\xff\xd9"
    decode = imgread.Loader().decode if persistent else imgread.load_numpy_from_bytes
    old = memory_policy.install(7 * 13 * 3)
    try:
        with pytest.warns(RuntimeWarning, match="turbojpeg decode failed"):
            fallback = decode(damaged)
        assert memory_policy.stats() == (1, 1, 0, 1)
        assert fallback.shape == (7, 13, 3) and not fallback.flags.owndata
    finally:
        memory_policy.restore(old)
    assert decode(data).flags.owndata


def test_allocator_callback_cannot_reenter_loader(memory_policy):
    data = encode(fmt="JPEG")
    loader = imgread.Loader()
    observed = []

    def callback():
        with pytest.raises(RuntimeError, match="Loader is busy"):
            loader.decode(data)
        observed.append("busy")

    old = memory_policy.install(7 * 13 * 3, callback)
    try:
        result = loader.decode(data)
        assert observed == ["busy"]
        assert memory_policy.stats() == (1, 0, 0, 1)
    finally:
        memory_policy.restore(old)
    np.testing.assert_array_equal(result, loader.decode(data))
