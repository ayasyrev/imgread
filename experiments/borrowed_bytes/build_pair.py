"""Build unchanged NumPy-output baseline and public borrowed-bytes candidate."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from pathlib import Path

BASE = "ee98d996d0388bef66f0182c4acfae20a146c81c"
ROOT = Path(__file__).resolve().parents[2]


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-prefix", type=Path)
    parser.add_argument("--nasm", type=Path, required=True)
    parser.add_argument("--cmake-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = output / "baseline-source"
    baseline.mkdir()
    archive = subprocess.check_output(["git", "archive", BASE], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(baseline, filter="data")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("TURBOJPEG_") or key in {"RUSTFLAGS", "CFLAGS", "LDFLAGS", "CC"}:
            del env[key]
    build_env = {
        "CARGO_TARGET_DIR": str(output / "target"),
        "CARGO_NET_OFFLINE": "true",
        "CARGO_BUILD_JOBS": "4",
        "PYO3_PYTHON": sys.executable,
        "CC": "/usr/bin/gcc",
        "CFLAGS": "-O3 -fno-lto",
        "LDFLAGS": "-fno-lto",
    }
    if args.native_prefix:
        prefix = args.native_prefix.resolve()
        build_env.update(
            TURBOJPEG_SOURCE="explicit", TURBOJPEG_DYNAMIC="1", TURBOJPEG_STATIC="0",
            TURBOJPEG_LIB_DIR=str(prefix / "lib"),
            TURBOJPEG_INCLUDE_DIR=str(prefix / "include"),
        )
        env["LD_LIBRARY_PATH"] = str(prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    else:
        build_env.update(TURBOJPEG_SOURCE="vendor", TURBOJPEG_STATIC="1", TURBOJPEG_DYNAMIC="0")
    env.update(build_env)
    env["PATH"] = os.pathsep.join((str(args.cmake_dir), str(args.nasm.parent), env["PATH"]))
    command = ["cargo", "build", "--release", "--locked", "--offline",
               "--features", "extension-module,turbojpeg"]
    record = {
        "baseline": BASE,
        "candidate_base": capture(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "candidate_patch": capture(["git", "diff", "--", "src/lib.rs"], cwd=ROOT),
        "python": sys.version,
        "rustc": capture(["rustc", "--version"]),
        "compiler": capture([build_env["CC"], "--version"]).splitlines()[0],
        "environment": build_env,
        "command": command,
        "variants": {},
    }
    for name, source in (("baseline", baseline), ("candidate", ROOT)):
        # Cargo can consider equal-name path packages fresh after switching
        # checkouts with an existing target. Retain dependencies, rebuild imgread.
        subprocess.run(["cargo", "clean", "--release", "--package", "imgread"],
                       cwd=source, env=env, check=True)
        subprocess.run(command, cwd=source, env=env, check=True)
        install = output / name
        shutil.copytree(source / "python/imgread", install / "imgread",
                        ignore=shutil.ignore_patterns("*.so", "__pycache__"))
        extension = install / "imgread" / ("_native" + sysconfig.get_config_var("EXT_SUFFIX"))
        shutil.copy2(output / "target/release/libimgread.so", extension)
        linked = capture(["ldd", str(extension)], env=env)
        if args.native_prefix and str(prefix / "lib/libturbojpeg.so.0") not in linked:
            raise RuntimeError(linked)
        if not args.native_prefix and "libturbojpeg.so" in linked:
            raise RuntimeError("Static build unexpectedly requires shared TurboJPEG")
        subprocess.run([sys.executable, "-c",
                        ("import imgread, sys; assert not imgread._native._debug_build; "
                        "assert 'turbojpeg' in imgread.supported_backends(); "
                        "assert ('Immutable bytes are borrowed' in "
                        "imgread.load_numpy_from_bytes.__doc__) == (sys.argv[1] == 'candidate')"),
                        name],
                       env={**env, "PYTHONPATH": str(install)}, cwd=output, check=True)
        record["variants"][name] = {"source": str(source), "install": str(install), "ldd": linked}
        (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Built {name}: {extension}", flush=True)


if __name__ == "__main__":
    main()
