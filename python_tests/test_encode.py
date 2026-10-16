"""Cache encoding contract and independent decoding of jpegli output."""
import io
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import imgread


@pytest.fixture
def rgb():
    # Unequal dimensions and channels catch shape, color and stride mistakes.
    y, x = np.mgrid[:39, :53]
    return np.stack((x * 3, y * 5, 230 - x - y), axis=-1).astype(np.uint8)


@pytest.mark.parametrize("subsampling", ["420", "422", "440", "444"])
@pytest.mark.parametrize("progressive", [False, 1, True, 3])
def test_roundtrip(rgb, subsampling, progressive):
    ajpegli = pytest.importorskip("ajpegli")
    data = imgread.encode_jpeg(
        rgb, quality=95, subsampling=subsampling, progressive=progressive,
    )
    assert isinstance(data, bytes)
    header = ajpegli.info(data)
    assert (header.height, header.width) == rgb.shape[:2]
    assert bool(header.progressive) == bool(progressive)
    with Image.open(io.BytesIO(data)) as reference:
        assert reference.format == "JPEG"
        expected = np.asarray(reference.convert("RGB"))
    loader = imgread.Loader()
    actual = loader.decode(data)
    assert actual.shape == rgb.shape and actual.dtype == np.uint8
    assert actual.flags.c_contiguous and actual.flags.writeable
    # Independent decoder, including builds with the portable Rust backend.
    assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 4
    assert np.abs(actual.astype(int) - rgb.astype(int)).mean() < 2
    np.testing.assert_array_equal(actual, imgread.load_numpy_from_bytes(data))
    saved = actual.copy()
    loader.decode(imgread.encode_jpeg(rgb[::-1], quality=50))
    del loader, data
    np.testing.assert_array_equal(actual, saved)


def test_bgr_strides_and_readonly_input(rgb):
    pytest.importorskip("ajpegli")
    rgb.flags.writeable = False
    expected = imgread.encode_jpeg(rgb, quality=90)
    assert imgread.encode_jpeg(rgb[..., ::-1], color="BGR", quality=90) == expected
    view = rgb[::-1, ::2]
    assert imgread.encode_jpeg(view) == imgread.encode_jpeg(np.ascontiguousarray(view))
    assert imgread.encode_jpeg(rgb, quality=np.int64(90), progressive=np.int64(0)) == expected


@pytest.mark.parametrize("image,exception", [
    (None, TypeError), ([[[1, 2, 3]]], TypeError),
    (np.zeros((4, 5, 3), dtype=np.float32), ValueError),
    (np.zeros((4, 5, 3), dtype=np.uint16), ValueError),
    (np.zeros((4, 5), dtype=np.uint8), ValueError),
    (np.zeros((4, 5, 4), dtype=np.uint8), ValueError),
    (np.zeros((0, 5, 3), dtype=np.uint8), ValueError),
    (np.zeros((1, 65501, 3), dtype=np.uint8), ValueError),
])
def test_invalid_pixels(image, exception):
    with pytest.raises(exception):
        imgread.encode_jpeg(image)


@pytest.mark.parametrize("options", [
    {"quality": 0}, {"quality": 101}, {"quality": 90.5}, {"quality": True},
    {"subsampling": "gray"}, {"subsampling": "4:2:0"},
    {"progressive": -1}, {"progressive": 4}, {"progressive": "false"},
    {"color": "rgba"}, {"color": None},
])
def test_invalid_options(rgb, options):
    with pytest.raises(ValueError):
        imgread.encode_jpeg(rgb, **options)


def test_encoder_is_optional():
    # Exercise a clean import and decoding with the optional dependency blocked,
    # even in CI jobs where jpegli is installed for the integration tests.
    subprocess.run([sys.executable, "-c", """
import sys
sys.modules['ajpegli'] = None
import io
import numpy as np
from PIL import Image
import imgread
data = io.BytesIO()
Image.new('RGB', (3, 4), (10, 80, 160)).save(data, format='JPEG')
assert imgread.Loader().decode(data.getvalue()).shape == (4, 3, 3)
try:
    imgread.encode_jpeg(np.zeros((4, 3, 3), np.uint8))
except ImportError as exc:
    assert 'imgread[jpegli]' in str(exc)
else:
    raise AssertionError('missing encoder must raise ImportError')
"""], check=True)
