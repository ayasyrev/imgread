"""Generate notices from the locked, target-specific release graph (requires cargo-about)."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def generate():
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--all-features", "--format-version", "1"], cwd=ROOT))
    package = next(p for p in metadata["packages"] if p["name"] == "turbojpeg-sys")
    vendor = Path(package["manifest_path"]).parent / "libjpeg-turbo"
    version = re.search(r"set\(VERSION ([\d.]+)\)", (vendor / "CMakeLists.txt").read_text())[1]
    files = {}
    for name in ("LICENSE.md", "README.ijg"):
        files[Path("licenses") / ("libjpeg-turbo-" + version) / name] = (vendor / name).read_bytes()
    files[Path("licenses/rust/THIRD_PARTY_LICENSES.txt")] = subprocess.check_output(
        ["cargo", "about", "generate", "--locked", "--all-features", "about.hbs"], cwd=ROOT)
    text = f'''# Third-party notices

imgread includes code from the locked Rust dependency graph for Linux x86_64
and macOS x86_64/arm64. Full texts and per-license crate/version attribution,
including build dependencies, are in `licenses/rust/THIRD_PARTY_LICENSES.txt`.
Generate them with `uv run python scripts/update_notices.py` after changing Cargo.lock.

The `turbojpeg-sys` {package["version"]} crate vendors **libjpeg-turbo {version}**,
which is statically linked in official wheels. The Rust wrapper's MIT/Unlicense
choice does not replace the native IJG, BSD-3-Clause and zlib license requirements.
The exact vendored license files are preserved in `licenses/libjpeg-turbo-{version}/`.
Upstream: https://github.com/libjpeg-turbo/libjpeg-turbo/tree/{version}

This software is based in part on the work of the Independent JPEG Group.

NumPy is an independently installed Python dependency and carries its own license
notices. It is not bundled in imgread. No additional non-system native shared
libraries are permitted in release wheels without a new inventory and notices.
'''
    files[Path("THIRD_PARTY_NOTICES.md")] = text.encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = generate()
    for relative, content in files.items():
        path = ROOT / relative
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit(f"Notices are stale: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print("Notices verified" if args.check else "Notices updated")


if __name__ == "__main__":
    main()
