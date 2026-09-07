"""Regression checks for source transfer and distribution validation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import sys
import tomllib

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from export_public import FILES, export
from repository_metadata import configure


def source_tree(root):
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    (root / "Cargo.toml").write_text('[package]\nname = "imgread"\nversion = "0.2.0"\n\n[dependencies]\n')
    (root / "pyproject.toml").write_text('[project]\nname = "imgread"\n\n[tool.maturin]\nmodule-name = "imgread._native"\n')
    return root


def test_export_excludes_private_history_and_generated_files(tmp_path):
    source = source_tree(tmp_path / "private")
    for name in (".git/config", "docs/plans/private.md", "docs/python-tooling.md", "docs/releasing.md",
                 "docs/status.md", "AGENTS.md", "CLAUDE.md", ".agents/config.md", ".codex/config.toml",
                 ".env", "conductor/tasks.md",
                 "python/imgread/_native.so", "python/__pycache__/cached.pyc"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private fixture\n")
    original = (source / "pyproject.toml").read_bytes()
    destination = tmp_path / "public"
    export(source, destination, "example/decoder")
    assert not (destination / "docs").exists()
    configure(destination, "example/decoder", check=True)
    assert (source / "pyproject.toml").read_bytes() == original
    manifest = (destination / "PUBLIC_SOURCE_SHA256SUMS").read_text().splitlines()
    assert len(manifest) == len(FILES)
    for line in manifest:
        digest, name = line.split("  ", 1)
        assert digest == hashlib.sha256((destination / name).read_bytes()).hexdigest()
    assert {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} == set(FILES) | {"PUBLIC_SOURCE_SHA256SUMS"}


def test_export_refuses_existing_destination_and_symlinks(tmp_path):
    source = source_tree(tmp_path / "private")
    with pytest.raises(ValueError, match="already exists"):
        export(source, source)
    (source / "LICENSE").unlink()
    (source / "LICENSE").symlink_to(source / "README.md")
    with pytest.raises(ValueError, match="linked"):
        export(source, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_repository_metadata_requires_matching_destination(tmp_path):
    source = source_tree(tmp_path / "source")
    with pytest.raises(ValueError, match="URLs must point"):
        configure(source, "example/decoder", check=True)
    configure(source, "example/decoder")
    original = (source / "Cargo.toml").read_bytes()
    configure(source, "example/decoder")
    assert (source / "Cargo.toml").read_bytes() == original
    configure(source, "example/new-decoder")
    configure(source, "example/new-decoder", check=True)
    assert tomllib.loads((source / "Cargo.toml").read_text())["package"]["name"] == "imgread"
    with pytest.raises(ValueError, match="URLs must point"):
        configure(source, "example/decoder", check=True)
    for invalid in ("https://github.com/example/repo", "example/repo.git", "../repo", 'example/repo"\n'):
        with pytest.raises(ValueError):
            configure(source, invalid)


def test_sdist_checker_accepts_own_source_and_rejects_missing_license(tmp_path):
    import io
    import tarfile

    spec = spec_from_file_location("check_artifacts", SCRIPTS / "check_artifacts.py")
    checker = module_from_spec(spec)
    spec.loader.exec_module(checker)
    metadata = "Name: imgread\nVersion: " + checker.EXPECTED_VERSION + "\nLicense-Expression: MIT\nRequires-Python: >=3.11,<3.15\n"
    metadata += "".join(f"License-File: {name}\n" for name in checker.REQUIRED_LICENSES)
    files = {name: (checker.ROOT / name).read_bytes() for name in checker.REQUIRED_LICENSES}
    files.update({name: b"fixture" for name in ("Cargo.lock", "Cargo.toml", "src/lib.rs", "python/imgread/__init__.pyi", "python/imgread/py.typed")})
    files["PKG-INFO"] = (metadata + "\n# imgread\n").encode()
    files["scripts/check_artifacts.py"] = (SCRIPTS / "check_artifacts.py").read_bytes()
    artifact = tmp_path / "source.tar.gz"

    def pack():
        with tarfile.open(artifact, "w:gz") as archive:
            for name, content in files.items():
                info = tarfile.TarInfo("imgread/" + name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))

    pack()
    checker.check(artifact)
    for name in (".venv/lib/package/LICENSE", ".pytest_cache/README.md", "docs/plans/private.md",
                 "docs/python-tooling.md", "docs/releasing.md", "docs/status.md",
                 "AGENTS.md", "CLAUDE.md", "conductor/tasks.md", ".agents/config.md", ".codex/config.toml"):
        files[name] = b"private fixture"
        pack()
        with pytest.raises(ValueError, match="internal"):
            checker.check(artifact)
        del files[name]
    files["LICENSE"] = b"MIT draft"
    pack()
    with pytest.raises(ValueError, match="license text differs"):
        checker.check(artifact)
    del files["LICENSE"]
    pack()
    with pytest.raises(ValueError, match="missing license text"):
        checker.check(artifact)


def test_macos_checker_excludes_only_the_declared_install_name(tmp_path, monkeypatch):
    import zipfile

    spec = spec_from_file_location("check_artifacts", SCRIPTS / "check_artifacts.py")
    checker = module_from_spec(spec)
    spec.loader.exec_module(checker)
    prefix = f"imgread-{checker.EXPECTED_VERSION}.dist-info/"
    metadata = f"Name: imgread\nVersion: {checker.EXPECTED_VERSION}\nLicense-Expression: MIT\nRequires-Python: >=3.11,<3.15\n"
    metadata += "".join(f"License-File: {name}\n" for name in checker.REQUIRED_LICENSES)
    artifact = tmp_path / f"imgread-{checker.EXPECTED_VERSION}-cp311-cp311-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr(prefix + "METADATA", metadata + "\n# imgread\n")
        for name in checker.REQUIRED_LICENSES:
            archive.writestr(prefix + "licenses/" + name, (checker.ROOT / name).read_bytes())
        archive.writestr("imgread/__init__.pyi", "fixture\n")
        archive.writestr("imgread/py.typed", "")
        archive.writestr("imgread/_native.cpython-311-darwin.so", b"\xcf\xfa\xed\xfefixture")

    own_name = "@rpath/imgread._native.cpython-311-darwin.so"
    install_names = [own_name]
    libraries = [own_name, "/usr/lib/libiconv.2.dylib", "/usr/lib/libSystem.B.dylib"]

    def otool(command, *, text):
        assert command[0] == "otool" and text
        header = str(command[2]) + ":\n"
        if command[1] == "-D":
            return header + "\n".join(install_names) + "\n"
        assert command[1] == "-L"
        return header + "".join(f"\t{name} (compatibility version 1.0.0, current version 1.0.0)\n" for name in libraries)

    monkeypatch.setattr(checker.subprocess, "check_output", otool)
    checker.check(artifact)
    for name in ("@rpath/libjpeg.dylib", "/opt/homebrew/lib/libjpeg.dylib"):
        libraries.append(name)
        with pytest.raises(ValueError, match="undeclared macOS dependency"):
            checker.check(artifact)
        libraries.pop()
    install_names.clear()
    with pytest.raises(ValueError, match="undeclared macOS dependency"):
        checker.check(artifact)
