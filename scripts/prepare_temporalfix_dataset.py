"""Prepare a licence-gated manifest for an existing MOT17 download."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from temporalfix.dataset_manifest import build_mot17_manifest

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted MOT17 directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accept-license",
        required=True,
        metavar="SPDX_ID",
        help="Must be exactly CC-BY-NC-SA-3.0 after reviewing official terms.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the tree and write hashes without copying dataset content."""
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    manifest = build_mot17_manifest(args.source, accepted_license=args.accept_license)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sys.stdout.write(
        f"wrote manifest for {manifest['file_count']} files to {args.output}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
