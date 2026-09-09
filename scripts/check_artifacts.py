"""Validate distributions, metadata, notices, native dependencies and release tags."""
import argparse
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile

FORBIDDEN = {"AGENTS.md", "CLAUDE.md", "conductor", ".git", ".venv", ".pytest_cache", ".uv-cache", ".agents", ".codex", "__pycache__"}
ATTRIBUTION = b"This software is based in part on the work of the Independent JPEG Group."
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"].replace("-beta.", "b").replace("-dev.", ".dev")
REQUIRED_LICENSES = ["LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/rust/THIRD_PARTY_LICENSES.txt", "licenses/libjpeg-turbo-3.1.0/LICENSE.md", "licenses/libjpeg-turbo-3.1.0/README.ijg"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(path, allow_local=False):
    wheel = path.suffix == ".whl"
    if wheel:
        with zipfile.ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        metadata_name = next(name for name in files if name.endswith(".dist-info/METADATA"))
        prefix = metadata_name.removesuffix("METADATA") + "licenses/"
    else:
        with tarfile.open(path) as archive:
            files = {member.name.split("/", 1)[1]: archive.extractfile(member).read()
                     for member in archive.getmembers() if member.isfile() and "/" in member.name}
        metadata_name, prefix = "PKG-INFO", ""
        for name in ("Cargo.lock", "Cargo.toml", "src/lib.rs", "python/imgread/__init__.pyi", "python/imgread/py.typed"):
            require(name in files, f"{path.name}: missing source {name}")
    metadata = BytesParser(policy=default).parsebytes(files[metadata_name])
    require(metadata["Name"] == "imgread", "wrong project name")
    require(metadata["Version"] == EXPECTED_VERSION, "wrong package version")
    require(metadata["License-Expression"] == "MIT", "missing MIT License-Expression")
    require(metadata["Requires-Python"].replace(" ", "") == ">=3.11,<3.15", "wrong Python support range")
    require("# imgread" in metadata.get_payload(), "README missing from metadata")
    license_fields = metadata.get_all("License-File", [])
    for name in REQUIRED_LICENSES:
        require(name in license_fields, f"missing License-File: {name}")
        require(prefix + name in files, f"missing license text: {name}")
        require(files[prefix + name] == (ROOT / name).read_bytes(), f"license text differs from source: {name}")
    require(ATTRIBUTION in files[prefix + "THIRD_PARTY_NOTICES.md"], "missing IJG attribution")
    for name, content in files.items():
        parts = PurePosixPath(name).parts
        require(not (set(parts) & FORBIDDEN), f"internal file in artifact: {name}")
        require("docs" not in parts, f"internal document: {name}")
        if not allow_local:
            # Assemble markers so this source file can pass its own sdist check.
            markers = [b"/" + part for part in (b"home/", b"Users/", b"github/workspace/")]
            require(not any(marker in content for marker in markers), f"absolute build path in {name}")
    if wheel:
        require("imgread/__init__.pyi" in files and "imgread/py.typed" in files, "missing typing files")
        native = [name for name in files if name.endswith((".so", ".dylib", ".pyd"))]
        require(len(native) == 1 and native[0].startswith("imgread/_native."), f"unexpected bundled native libraries: {native}")
        if not allow_local:
            require(re.search(r"-cp31[1-4]-cp31[1-4]-", path.name), "unsupported interpreter/ABI tag")
            require(re.search(r"(manylinux_2_17_x86_64|manylinux2014_x86_64|macosx_10_13_x86_64|macosx_11_0_arm64)", path.name), "incorrect release platform tag")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "extension.so"
            binary.write_bytes(files[native[0]])
            if files[native[0]].startswith(b"\x7fELF"):
                dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
                dependencies = re.findall(r"Shared library: \[(.*?)\]", dynamic)
                permitted = {"libgcc_s.so.1", "libm.so.6", "libc.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1", "ld-linux-x86-64.so.2"}
                require(set(dependencies) <= permitted, f"undeclared native dependencies: {dependencies}")
                if not allow_local:
                    versions = subprocess.check_output(["readelf", "--version-info", str(binary)], text=True)
                    glibc = [tuple(map(int, match)) for match in re.findall(r"GLIBC_(\d+)\.(\d+)", versions)]
                    require(max(glibc, default=(0, 0)) <= (2, 17), "glibc exceeds 2.17")
            else:
                # otool -L includes LC_ID_DYLIB as well as imported libraries.
                # Exclude only the object's own install name reported by -D.
                install_names = {
                    line.strip() for line in subprocess.check_output(
                        ["otool", "-D", str(binary)], text=True).splitlines()[1:]
                }
                libraries = subprocess.check_output(["otool", "-L", str(binary)], text=True).splitlines()[1:]
                dependencies = [line.strip().split(" (compatibility version ", 1)[0] for line in libraries]
                dependencies = [name for name in dependencies if name not in install_names]
                require(all(name.startswith(("/usr/lib/", "/System/Library/")) for name in dependencies), f"undeclared macOS dependency: {dependencies}")
    print(f"Verified {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--allow-local", action="store_true", help="skip release platform and build-path checks for developer builds")
    args = parser.parse_args()
    for path in args.artifacts:
        check(path, args.allow_local)


if __name__ == "__main__":
    main()
