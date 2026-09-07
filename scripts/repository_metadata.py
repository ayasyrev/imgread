"""Set or check public package URLs for a GitHub OWNER/REPOSITORY."""
import argparse
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def repository_url(repository):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", repository):
        raise ValueError("expected GitHub OWNER/REPOSITORY, without URL or .git suffix")
    if repository.endswith(".git"):
        raise ValueError("omit the .git suffix")
    return "https://github.com/" + repository


def configure(root, repository, *, check=False):
    url = repository_url(repository)
    expected = {
        "Cargo.toml": ("package", {"repository": url, "homepage": url}),
        "pyproject.toml": ("project.urls", {"Homepage": url, "Repository": url, "Issues": url + "/issues"}),
    }
    updates = {}
    for filename, (section, values) in expected.items():
        path = root / filename
        content = path.read_text()
        data = tomllib.loads(content)
        actual = data
        for key in section.split("."):
            actual = actual.get(key, {})
        if check:
            if any(actual.get(key) != value for key, value in values.items()):
                raise ValueError(f"{filename}: URLs must point to {url}")
            continue
        header = f"[{section}]"
        pattern = re.compile(r"(?m)^" + re.escape(header) + r"\s*\n(.*?)(?=^\[|\Z)", re.S)
        match = pattern.search(content)
        body = match.group(1) if match else ""
        for key in values:
            body = re.sub(r"(?m)^" + re.escape(key) + r"\s*=.*\n?", "", body)
        body = body.rstrip() + "\n" if body.strip() else ""
        body += "".join(f'{key} = "{value}"\n' for key, value in values.items()) + "\n"
        replacement = header + "\n" + body
        content = content[:match.start()] + replacement + content[match.end():] if match else content.rstrip() + "\n\n" + replacement
        tomllib.loads(content)
        updates[path] = content
    for path, content in updates.items():
        path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", help="GitHub OWNER/REPOSITORY of the new public repository")
    parser.add_argument("--check", action="store_true", help="verify URLs without modifying files")
    args = parser.parse_args()
    configure(ROOT, args.repository, check=args.check)
    print("Public repository metadata verified" if args.check else "Public repository metadata configured")


if __name__ == "__main__":
    main()
