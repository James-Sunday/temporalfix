"""Validate TemporalFix versions and licences inside built artifacts."""

from __future__ import annotations

import argparse
import tarfile
import tomllib
import zipfile
from pathlib import Path


def _project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _wheel(path: Path, version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        licences = [name for name in names if name.endswith("/licenses/LICENSE")]
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(licences) != 1 or len(archive.read(licences[0])) < 10_000:
            raise RuntimeError(f"{path.name} lacks the complete Apache-2.0 licence")
        if len(metadata) != 1:
            raise RuntimeError(f"{path.name} lacks one METADATA file")
        contents = archive.read(metadata[0]).decode()
        if f"Version: {version}\n" not in contents:
            raise RuntimeError(f"{path.name} metadata version does not match {version}")
        if "License-Expression: Apache-2.0\n" not in contents:
            raise RuntimeError(f"{path.name} lacks the Apache-2.0 expression")


def _source(path: Path, version: str) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        licences = [
            item for item in archive.getmembers() if item.name.endswith("/LICENSE")
        ]
        if len(licences) != 1 or licences[0].size < 10_000:
            raise RuntimeError(f"{path.name} lacks one complete Apache-2.0 licence")
        metadata = [
            item for item in archive.getmembers() if item.name.endswith("/PKG-INFO")
        ]
        if len(metadata) != 1:
            raise RuntimeError(f"{path.name} lacks one PKG-INFO file")
        extracted = archive.extractfile(metadata[0])
        if (
            extracted is None
            or f"Version: {version}\n" not in extracted.read().decode()
        ):
            raise RuntimeError(f"{path.name} metadata version does not match {version}")


def main() -> int:
    """Require exactly one matching wheel and source distribution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = _project_version(root)
    wheel_version = version.replace("-", "_")
    wheels = list(arguments.directory.glob(f"temporalfix-{wheel_version}-*.whl"))
    sources = list(arguments.directory.glob(f"temporalfix-{version}.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise RuntimeError(f"expected one wheel and sdist for temporalfix {version}")
    _wheel(wheels[0], version)
    _source(sources[0], version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
