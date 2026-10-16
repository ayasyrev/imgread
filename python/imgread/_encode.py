"""Optional jpegli encoding for caches consumed by Loader.decode."""
from operator import index

import numpy as np
from numpy.typing import NDArray


def encode_jpeg(
    image: NDArray[np.uint8],
    *,
    quality: int = 90,
    subsampling: str = "420",
    progressive: bool | int = False,
    color: str = "rgb",
) -> bytes:
    """Encode HWC uint8 RGB/BGR pixels to JPEG bytes using optional jpegli.

    Install ``imgread[jpegli]`` to encode. Decoding the result needs only imgread.
    Sequential JPEG (progressive=False) is the default for fast cache reads.
    True selects jpegli's progressive level 2; integers 0..3 select a level.
    Quality is 1..100 (100 is still lossy); subsampling is 420/422/440/444.
    Strided input and BGR input are copied to contiguous RGB before encoding.
    No dtype conversion, alpha removal, orientation or ICC transform is applied.
    The caller must not mutate the input while encoding is in progress.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if not all(0 < size <= 65500 for size in image.shape[:2]):
        raise ValueError("JPEG width and height must be between 1 and 65500")
    if not isinstance(color, str) or color.lower() not in ("rgb", "bgr"):
        raise ValueError("color must be 'rgb' or 'bgr'")
    if subsampling not in ("420", "422", "440", "444"):
        raise ValueError("subsampling must be '420', '422', '440' or '444'")
    if isinstance(quality, (bool, np.bool_)):
        raise ValueError("quality must be an integer between 1 and 100")
    try:
        quality = index(quality)
    except TypeError as exc:
        raise ValueError("quality must be an integer between 1 and 100") from exc
    if not 1 <= quality <= 100:
        raise ValueError("quality must be an integer between 1 and 100")
    if isinstance(progressive, (bool, np.bool_)):
        progressive = 2 if progressive else 0
    else:
        try:
            progressive = index(progressive)
        except TypeError as exc:
            raise ValueError("progressive must be a bool or an integer from 0 to 3") from exc
        if not 0 <= progressive <= 3:
            raise ValueError("progressive must be a bool or an integer from 0 to 3")

    # Loading imgread and decoding existing caches never imports the encoder.
    try:
        import ajpegli
    except ModuleNotFoundError as exc:
        if exc.name != "ajpegli":
            raise
        raise ImportError("JPEG encoding requires the optional 'imgread[jpegli]' extra") from exc

    rgb = np.ascontiguousarray(image[..., ::-1] if color.lower() == "bgr" else image)
    try:
        return ajpegli.encode(
            rgb, quality=quality, subsampling=subsampling, progressive=progressive,
            mode="RGB", dtype="uint8", allow_copy=False, alpha="error",
            adaptive_quantization=True, optimize=True, xyb=False,
        )
    except ajpegli.EncodeError as exc:
        raise RuntimeError(f"jpegli encoding failed: {exc}") from exc
