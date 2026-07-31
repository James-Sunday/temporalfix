"""Run the reproducible TemporalFix synthetic scaling benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from temporalfix.benchmarking import run_synthetic_scaling

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the benchmark argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[10, 50, 100, 500],
        help="Detection counts to measure (default: 10 50 100 500).",
    )
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="balanced")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_artifacts/temporalfix/synthetic-scaling.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Measure all requested counts and write a machine-readable artifact."""
    args = build_parser().parse_args(argv)
    result = run_synthetic_scaling(
        counts=args.counts,
        frames=args.frames,
        warmup_frames=args.warmup_frames,
        seed=args.seed,
        preset=args.preset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.stdout.write(f"wrote measured benchmark artifact to {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
