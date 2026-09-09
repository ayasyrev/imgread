import inspect
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import imgread
from test_beta_contract import encode
from test_resource_limits import oversized


@pytest.fixture
def paths(tmp_path):
    result = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (13, 7), (index, 50, 100)).save(path)
        result.append(path)
    return result


def test_signature_and_exports():
    signature = inspect.signature(imgread.Loader)
    assert str(signature) == "(paths=None, *, color='rgb', dtype='uint8', backend='auto', limits='safe', max_buffer_bytes=1048576)"
    assert "Loader" in imgread.__all__
    assert imgread.Loader.__module__ == "imgread"
    loader = imgread.Loader()
    for name in ("color", "dtype", "backend", "limits", "paths", "max_buffer_bytes"):
        with pytest.raises(AttributeError):
            setattr(loader, name, None)
    assert not hasattr(loader, "__dict__")


def test_empty_and_no_snapshot(paths):
    for loader in (imgread.Loader(), imgread.Loader(None)):
        assert bool(loader)
        with pytest.raises(TypeError):
            len(loader)
        with pytest.raises(TypeError):
            loader[0]
        assert loader(paths[0]).shape == (7, 13, 3)
    loader = imgread.Loader([])
    assert len(loader) == 0 and bool(loader)
    with pytest.raises(IndexError):
        loader[0]
    assert loader(paths[0]).shape == (7, 13, 3)


def test_snapshot_once_mutation_order_and_duplicates(paths):
    class MutablePath:
        calls = 0
        path = paths[1]
        def __fspath__(self):
            self.calls += 1
            return str(self.path)
    value = MutablePath()
    source = [paths[2], value, paths[2]]
    loader = imgread.Loader(item for item in source)
    source.clear()
    value.path = paths[0]
    assert value.calls == 1 and len(loader) == 3
    assert [int(loader[i][0, 0, 0]) for i in range(3)] == [2, 1, 2]
    assert value.calls == 1
    assert loader(paths[0])[0, 0, 0] == 0


@pytest.mark.parametrize("bad", ["x", b"x", Path("x"), 1, [None], [b"x"], [1]])
def test_invalid_snapshot(bad):
    with pytest.raises(TypeError):
        imgread.Loader(bad)


def test_iterable_pathlike_rejected_before_iteration():
    class PathIterable:
        def __fspath__(self):
            raise AssertionError("must not convert container")
        def __iter__(self):
            raise AssertionError("must not iterate container")
    with pytest.raises(TypeError):
        imgread.Loader(PathIterable())


def test_options_before_iteration():
    def paths():
        raise AssertionError("iterated invalid configuration")
        yield "never"
    for key in ("color", "dtype", "backend", "limits"):
        with pytest.raises(ValueError):
            imgread.Loader(paths(), **{key: "bad"})
    for cap in (-1, 10**100, -(10**100)):
        with pytest.raises(ValueError):
            imgread.Loader(max_buffer_bytes=cap)
    for cap in (None, 1.5, "1", []):
        with pytest.raises(TypeError):
            imgread.Loader(max_buffer_bytes=cap)
    for cap in (False, True, 0, 1, np.int64(1024)):
        assert imgread.Loader(max_buffer_bytes=cap)
    class Cap:
        calls = 0
        def __index__(self):
            self.calls += 1
            return 0
    cap = Cap()
    imgread.Loader(max_buffer_bytes=cap)
    assert cap.calls == 1


@pytest.mark.parametrize("index,expected", [(0, 0), (2, 2), (-1, 2), (-3, 0), (np.int64(-2), 1), (np.uint64(1), 1)])
def test_indices(paths, index, expected):
    assert imgread.Loader(paths)[index][0, 0, 0] == expected


@pytest.mark.parametrize("index", [3, -4, 10**100, -(10**100), np.uint64(2**64 - 1)])
def test_out_of_bounds(paths, index):
    with pytest.raises(IndexError):
        imgread.Loader(paths)[index]


@pytest.mark.parametrize("index", [True, False, slice(None), [0], {0: 0}, 0.0, "0", None])
def test_index_types(paths, index):
    with pytest.raises(TypeError):
        imgread.Loader(paths)[index]


def test_index_protocol_once(paths):
    class Index:
        calls = 0
        def __index__(self):
            self.calls += 1
            return -1
    index = Index()
    assert imgread.Loader(paths)[index][0, 0, 0] == 2
    assert index.calls == 1


def test_lazy_files_cwd_unicode_and_surrogates(tmp_path, monkeypatch):
    loader = imgread.Loader(["relative.png"])
    for directory, color in ((tmp_path / "a", 1), (tmp_path / "b", 2)):
        directory.mkdir()
        monkeypatch.chdir(directory)
        Image.new("RGB", (2, 3), (color, 0, 0)).save("relative.png")
        assert loader[0][0, 0, 0] == color
        Image.new("RGB", (2, 3), (color + 3, 0, 0)).save("relative.png")
        assert loader[0][0, 0, 0] == color + 3
        Path("relative.png").unlink()
        with pytest.raises(FileNotFoundError):
            loader[0]
    for name in ("картинка.png", os.fsdecode(b"image-\xff.png")):
        path = tmp_path / name
        path.write_bytes(encode())
        np.testing.assert_array_equal(imgread.Loader([path])[0], imgread.load_numpy(path))


CASES = [("RGB", "JPEG", {}), ("RGB", "JPEG", {"progressive": True}),
         ("RGB", "JPEG", {"optimize": True}), ("L", "JPEG", {}), ("CMYK", "JPEG", {}),
         ("L", "JPEG", {"progressive": True}), ("CMYK", "JPEG", {"progressive": True}),
         ("RGB", "PNG", {}), ("RGBA", "PNG", {}), ("L", "PNG", {}), ("I;16", "PNG", {}),
         ("RGB", "TIFF", {"compression": "raw"}), ("RGB", "TIFF", {"compression": "tiff_lzw"}),
         ("RGB", "TIFF", {"compression": "tiff_adobe_deflate"}), ("I;16", "TIFF", {})]


@pytest.mark.parametrize("mode,fmt,options", CASES)
@pytest.mark.parametrize("backend", ["auto", "image", "turbojpeg"])
@pytest.mark.parametrize("color", ["rgb", "bgr"])
@pytest.mark.parametrize("limits", ["safe", "unlimited"])
def test_exact_function_parity(mode, fmt, options, backend, color, limits, tmp_path):
    path = tmp_path / "fixture"
    path.write_bytes(encode(mode, fmt, **options))
    loader = imgread.Loader([path], backend=backend, color=color.upper(), limits=limits.upper(), dtype="UINT8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = imgread.load_numpy(path, backend=backend, color=color, limits=limits)
        for actual in (loader(path), loader[0], loader.decode(path.read_bytes()), loader[0]):
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.uint8 and actual.flags.writeable and actual.flags.c_contiguous


def test_tiff_first_page_and_endian(tmp_path):
    path = tmp_path / "pages.tiff"
    for mode in ("I;16", "I;16B"):
        Image.new(mode, (4, 3), 32768).save(path)
        np.testing.assert_array_equal(imgread.Loader([path])[0], imgread.load_numpy(path))
    Image.new("RGB", (3, 2), (1, 2, 3)).save(path, save_all=True, append_images=[Image.new("RGB", (8, 9))])
    assert imgread.Loader([path])[0].shape == (2, 3, 3)


def test_output_independence_after_errors_and_deletion(paths):
    loader = imgread.Loader(paths)
    outputs = [loader[0] for _ in range(8)]
    expected = outputs[0].copy()
    with pytest.raises(FileNotFoundError):
        loader("does-not-exist")
    loader[2]
    del loader
    outputs[0][:] = 255
    for result in outputs[1:]:
        assert not np.shares_memory(result, outputs[0])
        np.testing.assert_array_equal(result, expected)


def outcome(call, path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pixels = call(path)
            result = ("ok", pixels.shape, pixels.tobytes())
        except Exception as error:
            result = (type(error), str(error), getattr(error, "errno", None), getattr(error, "filename", None))
        return result, [(item.category, str(item.message)) for item in caught]


def test_ordered_failures_and_recovery(tmp_path):
    files = []
    for name, data in [("good.jpg", encode(fmt="JPEG")), ("corrupt.jpg", b"\xff\xd8\xff\x00"),
                       ("huge.jpg", oversized("JPEG")), ("fallback.png", encode())]:
        path = tmp_path / name
        path.write_bytes(data)
        files.append(path)
    loader = imgread.Loader(backend="turbojpeg")
    for path in [files[0], files[1], files[2], tmp_path / "missing", tmp_path, files[3], files[0]]:
        assert outcome(loader, path) == outcome(lambda p: imgread.load_numpy(p, backend="turbojpeg"), path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning):
            loader(files[3])
    assert outcome(loader, files[0]) == outcome(lambda p: imgread.load_numpy(p, backend="turbojpeg"), files[0])


@pytest.mark.parametrize("policy,count", [("always", 3), ("default", 1), ("ignore", 0)])
def test_warning_filters_and_location(tmp_path, policy, count):
    path = tmp_path / "image.png"
    path.write_bytes(encode())
    loader = imgread.Loader(backend="turbojpeg")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        for _ in range(3):
            line = inspect.currentframe().f_lineno + 1
            loader(path)
    assert len(caught) == count
    for item in caught:
        assert item.category is RuntimeWarning and item.filename == __file__ and item.lineno == line
