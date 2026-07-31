"""Create a dependency-free SVG latency plot from a measured artifact."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def render_svg(artifact: dict[str, Any]) -> str:
    """Render median and P95 latency values into a simple SVG."""
    if artifact.get("benchmark") != "temporalfix.synthetic_scaling":
        raise ValueError("not a TemporalFix synthetic-scaling artifact")
    metrics = artifact.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("artifact contains no metrics")
    width, height = 720, 420
    left, top, plot_width, plot_height = 70, 35, 610, 320
    counts = [int(item["detections_per_frame"]) for item in metrics]
    medians = [float(item["latency_ms"]["median"]) for item in metrics]
    p95 = [float(item["latency_ms"]["p95"]) for item in metrics]
    maximum = max(p95) or 1.0
    step = plot_width / max(1, len(metrics) - 1)

    def points(values: list[float]) -> str:
        return " ".join(
            f"{left + index * step:.1f},{top + plot_height * (1 - value / maximum):.1f}"
            for index, value in enumerate(values)
        )

    labels = []
    for index, count in enumerate(counts):
        x = left + index * step
        labels.append(f'<text x="{x:.1f}" y="380" text-anchor="middle">{count}</text>')
    environment = html.escape(str(artifact.get("environment", {})))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="black"/>'
        f'<line x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>'
        f'<polyline points="{points(p95)}" fill="none" stroke="#d55e00" '
        'stroke-width="2"/>'
        f'<polyline points="{points(medians)}" fill="none" stroke="#0072b2" '
        'stroke-width="2"/>'
        f'<text x="10" y="20">Latency ms (max P95 {maximum:.3f})</text>'
        '<text x="300" y="410">Detections per frame</text>'
        '<text x="530" y="20" fill="#0072b2">Median</text>'
        '<text x="610" y="20" fill="#d55e00">P95</text>'
        f"<title>{environment}</title>{''.join(labels)}</svg>"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write an SVG derived only from an existing JSON artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    parsed = json.loads(args.artifact.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("artifact root must be an object")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(parsed), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
