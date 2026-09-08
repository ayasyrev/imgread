"""Deterministic read-only corpus selection and versioned synthetic inputs."""
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def select(config):
    root = Path(config["root"])
    classes = sorted(path for path in root.iterdir() if path.is_dir())
    if len(classes) != config["classes"]:
        raise ValueError("corpus class count changed")
    rows = []
    samples = []
    for label, directory in enumerate(classes):
        paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg"))
        if len(paths) < config["per_class"]:
            raise ValueError("corpus class is incomplete")
        for path in paths[:config["per_class"]]:
            relative = path.relative_to(root).as_posix()
            if "\t" in relative or "\n" in relative:
                raise ValueError("unrepresentable TSV path")
            rows.append(f"{relative}\t{path.stat().st_size}\t{digest(path)}\n")
            samples.append((str(path), label))
    manifest = "".join(rows).encode()
    if hashlib.sha256(manifest).hexdigest() != config["manifest_sha256"]:
        raise ValueError("corpus digest changed")
    if sum(int(row.split("\t")[1]) for row in rows) != config["compressed_bytes"]:
        raise ValueError("corpus compressed size changed")
    return samples, manifest


def samples_from_manifest(config, manifest):
    data = Path(manifest).read_bytes()
    if hashlib.sha256(data).hexdigest() != config["manifest_sha256"]:
        raise ValueError("saved corpus manifest mismatch")
    root = Path(config["root"])
    names = sorted({row.split("/", 1)[0] for row in data.decode().splitlines()})
    result = []
    for row in data.decode().splitlines():
        relative, _size, _sha = row.split("\t")
        result.append((str(root / relative), names.index(relative.split("/", 1)[0])))
    return result


def generate(config, directory):
    import numpy as np
    from PIL import Image, ImageFile
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    rng = np.random.default_rng(config["seed"])
    options = {"quality": config["jpeg_quality"], "subsampling": config["subsampling"]}
    small = rng.integers(0, 256, config["small_shape"], dtype=np.uint8)
    Image.fromarray(small).save(directory / "small.jpg", **options)
    large = rng.integers(0, 256, config["large_shape"], dtype=np.uint8)
    Image.fromarray(large).save(directory / "large.jpg", **options)
    # Pillow's two-bytes/pixel progressive output estimate is too small for
    # random high-quality 4:4:4 pixels. Supply sufficient encoder scratch space;
    # this affects generation only, not the fixed pixels or JPEG settings.
    previous_block = ImageFile.MAXBLOCK
    try:
        ImageFile.MAXBLOCK = max(previous_block, large.nbytes * 2 + 65536)
        Image.fromarray(large).save(directory / "progressive.jpg", progressive=True, **options)
    finally:
        ImageFile.MAXBLOCK = previous_block
    del large
    data = bytearray((directory / "large.jpg").read_bytes())
    scan = data.index(b"\xff\xda")
    entropy_start = scan + 2 + int.from_bytes(data[scan + 2:scan + 4], "big")
    data[entropy_start + 16:entropy_start + 20] = b"\xff\xc4\x00\x01"
    (directory / "corrupt.jpg").write_bytes(data)
    data = bytearray((directory / "small.jpg").read_bytes())
    frame = data.index(b"\xff\xc0")
    data[frame + 5:frame + 9] = struct.pack(">HH", 65000, 65000)
    (directory / "oversized.jpg").write_bytes(data)
    rows = {path.stem: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(directory.glob("*.jpg"))}
    if rows["large"]["bytes"] <= 1048576 or rows["progressive"]["bytes"] <= 1048576:
        raise ValueError("large stress images must exceed retention cap")
    return {"recipe": config, "inputs": rows}


def manifest_paths(count):
    # 9 + 6 + 1 + 44 + 4 = 64 ASCII bytes, including all one million indices.
    return [f"manifest/{index:06d}/" + "x" * 44 + ".jpg" for index in range(count)]
