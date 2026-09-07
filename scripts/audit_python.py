"""Audit all uv.lock entries through a temporary standard PEP 751 lockfile."""
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    subprocess.run([
        "uv", "export", "--locked", "--all-groups", "--no-emit-project",
        "--format", "pylock.toml", "--output-file", str(Path(directory) / "pylock.toml"),
    ], cwd=root, stdout=subprocess.DEVNULL, check=True)
    subprocess.run([sys.executable, "-m", "pip_audit", "--locked", directory], check=True)
