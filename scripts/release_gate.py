"""Reject incomplete metadata and unsafe target/version combinations."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

PLACEHOLDERS = (
    "OWNER_PLACEHOLDER",
    "PACKAGE_OWNER_PLACEHOLDER",
    "SECURITY_CONTACT_PLACEHOLDER",
)


def main() -> int:
    """Validate repository metadata immediately before an external publish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("testpypi", "pypi"), required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        version = str(tomllib.load(handle)["project"]["version"])

    text_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() in {".cff", ".md", ".toml", ".yml", ".yaml"}
    ]
    unresolved = [
        path.relative_to(root)
        for path in text_files
        if any(marker in path.read_text(encoding="utf-8") for marker in PLACEHOLDERS)
    ]
    if unresolved:
        joined = ", ".join(str(path) for path in unresolved)
        raise RuntimeError(f"unresolved publication placeholders: {joined}")
    if arguments.target == "pypi" and any(
        marker in version for marker in ("a", "b", "rc", ".dev")
    ):
        raise RuntimeError(
            f"production PyPI requires a stable version, found {version}"
        )
    print(f"release gate passed for {arguments.target}: temporalfix {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
