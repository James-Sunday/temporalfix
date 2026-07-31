"""Licence-gated manifests for external benchmark datasets."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MOT17_LICENSE = "CC-BY-NC-SA-3.0"
_MOT17_SOURCE = "https://motchallenge.net/data/MOT17Det/"
_CHUNK_BYTES = 1024 * 1024


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_mot17(files: list[Path], root: Path) -> None:
    relative = {path.relative_to(root).as_posix() for path in files}
    has_sequence_info = any(path.endswith("/seqinfo.ini") for path in relative)
    has_annotations = any(
        path.endswith(("/det/det.txt", "/gt/gt.txt")) for path in relative
    )
    if not has_sequence_info or not has_annotations:
        raise ValueError(
            "source does not look like extracted MOT17 data: expected seqinfo.ini "
            "and det/det.txt or gt/gt.txt"
        )


def build_mot17_manifest(
    source: str | Path, *, accepted_license: str
) -> dict[str, Any]:
    """Hash an existing MOT17 tree after explicit licence acceptance.

    No local absolute path is stored in the returned manifest.
    """
    if accepted_license != _MOT17_LICENSE:
        raise ValueError(
            f"licence acceptance must exactly equal {_MOT17_LICENSE!r}; "
            "review the official terms before continuing"
        )
    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError(f"source is not a directory: {source}")

    paths = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError("dataset source must not contain symbolic links")
    files = [path for path in paths if path.is_file()]
    if not files:
        raise ValueError("dataset source contains no files")
    _validate_mot17(files, root)

    records = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = _file_digest(path)
        record = {"path": relative, "size_bytes": size, "sha256": digest}
        records.append(record)
        total_bytes += size
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode())

    return {
        "schema_version": 1,
        "dataset": "MOT17",
        "source_url": _MOT17_SOURCE,
        "license": _MOT17_LICENSE,
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "prepared_at": datetime.now(UTC).isoformat(),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "manifest_sha256": aggregate.hexdigest(),
        "files": records,
        "notes": [
            "The dataset is not bundled or redistributed by TemporalFix.",
            "MOTChallenge requires attribution, non-commercial use and "
            "share-alike for adapted dataset material.",
            "Cite MOTChallenge and the original sequence sources listed on its "
            "data page.",
        ],
    }
