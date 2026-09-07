"""Export a reviewable source snapshot for a new public repository, without Git history."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

from repository_metadata import configure

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ".gitignore", "Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock",
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "about.toml", "about.hbs", "deny.toml",
    "python/imgread/py.typed",
    "python_tests/fixtures/README.md",
    ".github/workflows/ci.yml", ".github/workflows/beta-build.yml", ".github/workflows/release.yml",
)
PATTERNS = ("src/**/*.rs", "tests/**/*.rs", "python/**/*.py", "python/**/*.pyi",
            "python_tests/**/*.py", "scripts/**/*.py", "licenses/**/*.md",
            "licenses/**/*.txt", "licenses/**/README.ijg")


def export(source, destination, repository=None):
    source, destination = source.resolve(), destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination already exists; choose a new directory")
    paths = {Path(name) for name in FILES}
    for pattern in PATTERNS:
        paths.update(path.relative_to(source) for path in source.glob(pattern))
    for relative in paths:
        path = source / relative
        if not path.is_file() or path.is_symlink() or path.resolve() != source / relative:
            raise ValueError(f"missing, linked or non-regular source: {relative}")
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts) and relative.as_posix() not in FILES:
            raise ValueError(f"unexpected hidden source: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".public-export-", dir=destination.parent) as directory:
        stage = Path(directory) / "source"
        for relative in sorted(paths):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        if repository:
            configure(stage, repository)
        # Hashes cover every exported file; this manifest is for reviewing the transfer.
        manifest = "".join(f"{hashlib.sha256((stage / path).read_bytes()).hexdigest()}  {path.as_posix()}\n" for path in sorted(paths))
        (stage / "PUBLIC_SOURCE_SHA256SUMS").write_text(manifest)
        stage.rename(destination)
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new directory; existing paths are never overwritten")
    parser.add_argument("--repository", help="optional GitHub OWNER/REPOSITORY; sets URLs only in the exported copy")
    args = parser.parse_args()
    count = export(ROOT, args.destination, args.repository)
    print(f"Exported {count} files to {args.destination}; review PUBLIC_SOURCE_SHA256SUMS before importing")
    if not args.repository:
        print("Repository URLs copied from source; use --repository to override them")


if __name__ == "__main__":
    main()
