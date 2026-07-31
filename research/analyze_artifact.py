"""Render a measured synthetic-scaling JSON artifact as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def render_markdown(artifact: dict[str, Any]) -> str:
    """Validate the expected schema and render measured values."""
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported artifact schema_version")
    if artifact.get("benchmark") != "temporalfix.synthetic_scaling":
        raise ValueError("not a TemporalFix synthetic-scaling artifact")
    lines = [
        "| Detections/frame | Median (ms) | P95 (ms) | Traced peak bytes |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for metric in artifact["metrics"]:
        latency = metric["latency_ms"]
        lines.append(
            f"| {metric['detections_per_frame']} | {latency['median']:.3f} | "
            f"{latency['p95']:.3f} | {metric['python_allocation_peak_bytes']} |"
        )
    lines.extend(
        [
            "",
            "These values describe only the environment and configuration stored "
            "in the source JSON artifact.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Read an artifact and write or print a derived table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    parsed = json.loads(args.artifact.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("artifact root must be an object")
    rendered = render_markdown(parsed)
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
