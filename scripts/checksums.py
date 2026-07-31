"""Write deterministic SHA-256 checksums for release distributions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Hash wheel and source-distribution files in stable filename order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or arguments.directory / "SHA256SUMS"
    artifacts = sorted(
        (
            *arguments.directory.glob("*.whl"),
            *arguments.directory.glob("*.tar.gz"),
        ),
        key=lambda item: item.name,
    )
    if not artifacts:
        raise RuntimeError("no wheel or source-distribution artifacts found")
    lines = [f"{_sha256(path)}  {path.name}" for path in artifacts]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
