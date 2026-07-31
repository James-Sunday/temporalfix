"""Command-line interface for TemporalFix."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from temporalfix._version import __version__
from temporalfix.config import TemporalFixConfig, load_config
from temporalfix.detections import Detections
from temporalfix.repairer import TemporalRepairer

MAX_PREDICTIONS_BYTES = 64 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    """Build the TemporalFix argument parser."""
    parser = argparse.ArgumentParser(
        prog="temporalfix",
        description="Repair and stabilize frame-level video detections.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect-config", help="Print all resolved values for a preset."
    )
    inspect.add_argument(
        "--preset",
        default="balanced",
        choices=(
            "balanced",
            "low_latency",
            "high_stability",
            "strict_false_positive_control",
        ),
    )
    inspect.add_argument("--format", choices=("yaml", "json"), default="yaml")

    validate = subparsers.add_parser(
        "validate-config", help="Validate YAML and print resolved values."
    )
    validate.add_argument("config", type=Path)

    process = subparsers.add_parser(
        "process-video",
        help="Repair detector-independent JSON predictions for a video.",
    )
    process.add_argument("input", type=Path, help="Existing source video path.")
    process.add_argument("--detections", required=True, type=Path)
    process.add_argument("--output", type=Path, default=Path("repaired.json"))
    process.add_argument("--config", type=Path)
    process.add_argument("--preset", default="balanced")

    benchmark = subparsers.add_parser(
        "benchmark", help="Run a configured synthetic benchmark."
    )
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument(
        "--output", type=Path, default=Path("temporalfix-benchmark.json")
    )

    subparsers.add_parser("version", help="Print the installed version.")
    return parser


def _write_config(config: TemporalFixConfig, output_format: str) -> None:
    if output_format == "json":
        sys.stdout.write(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(yaml.safe_dump(config.to_dict(), sort_keys=True))


def _load_predictions(path: Path) -> list[Mapping[str, Any]]:
    if path.stat().st_size > MAX_PREDICTIONS_BYTES:
        raise ValueError(f"detections file exceeds {MAX_PREDICTIONS_BYTES} bytes")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    frames = parsed.get("frames") if isinstance(parsed, Mapping) else parsed
    if not isinstance(frames, list) or not all(
        isinstance(item, Mapping) for item in frames
    ):
        raise ValueError(
            "detections JSON must be a list or an object with a frames list"
        )
    return frames


def _process(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        raise ValueError(f"input video does not exist: {args.input}")
    config = (
        load_config(args.config)
        if args.config
        else TemporalFixConfig.preset(args.preset)
    )
    frames = _load_predictions(args.detections)
    repairer = TemporalRepairer(config)
    results: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        raw = frame.get("detections", frame)
        if not isinstance(raw, Mapping):
            raise ValueError(f"frame {index} detections must be an object")
        timestamp_value = frame.get("timestamp")
        timestamp = None if timestamp_value is None else float(timestamp_value)
        stream_id = str(frame.get("stream_id", "default"))
        fixed = repairer.update(
            Detections.from_dict(raw),
            timestamp=timestamp,
            stream_id=stream_id,
        )
        results.append(
            {
                "frame_index": index,
                "timestamp": timestamp,
                "stream_id": stream_id,
                "detections": fixed.to_dict(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema_version": 1, "frames": results}, indent=2),
        encoding="utf-8",
    )
    sys.stdout.write(f"wrote {len(results)} repaired frames to {args.output}\n")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    if not args.config.is_file():
        raise ValueError(f"benchmark configuration does not exist: {args.config}")
    parsed = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, Mapping):
        raise TypeError("benchmark configuration root must be a mapping")
    allowed = {
        "preset",
        "repair_config",
        "frames",
        "warmup_frames",
        "detections_per_frame",
        "seed",
    }
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"unknown benchmark options: {', '.join(unknown)}")
    frames = int(parsed.get("frames", 100))
    warmup = int(parsed.get("warmup_frames", 10))
    count = int(parsed.get("detections_per_frame", 10))
    seed = int(parsed.get("seed", 0))
    if frames < 1 or warmup < 0 or count < 0:
        raise ValueError("frames must be positive; warmup/count must be non-negative")
    preset = TemporalFixConfig.preset(str(parsed.get("preset", "balanced")))
    overrides = parsed.get("repair_config", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("repair_config must be a mapping")
    config = TemporalFixConfig.from_dict({**preset.to_dict(), **overrides})
    repairer = TemporalRepairer(config)
    generator = np.random.default_rng(seed)
    base = np.arange(count, dtype=np.float64) * 20.0

    def benchmark_frame() -> Detections:
        jitter = generator.normal(0.0, 0.25, size=count)
        x1 = base + jitter
        boxes = np.column_stack((x1, np.zeros(count), x1 + 10.0, np.full(count, 10.0)))
        return Detections(
            boxes,
            confidence=np.full(count, 0.9),
            class_id=np.zeros(count, dtype=np.int64),
        )

    for _ in range(warmup):
        repairer.update(benchmark_frame())
    repairer.reset()
    latencies = []
    for _ in range(frames):
        started = time.perf_counter_ns()
        repairer.update(benchmark_frame())
        latencies.append(time.perf_counter_ns() - started)
    samples = np.asarray(latencies, dtype=np.float64) / 1_000_000.0
    result = {
        "schema_version": 1,
        "benchmark": "temporalfix.synthetic",
        "configuration": {
            "frames": frames,
            "warmup_frames": warmup,
            "detections_per_frame": count,
            "seed": seed,
            "repair_config": config.to_dict(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "latency_ms": {
            "median": float(np.median(samples)),
            "p95": float(np.percentile(samples, 95)),
            "samples": samples.tolist(),
        },
        "limitations": [
            "Synthetic boxes do not measure detector accuracy.",
            "Memory measurement is introduced in the Phase 3 benchmark suite.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.stdout.write(f"wrote {frames} measured synthetic frames to {args.output}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TemporalFix command line with non-zero real-failure exits."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            sys.stdout.write(f"temporalfix {__version__}\n")
            return 0
        if args.command == "inspect-config":
            _write_config(TemporalFixConfig.preset(args.preset), args.format)
            return 0
        if args.command == "validate-config":
            _write_config(load_config(args.config), "yaml")
            return 0
        if args.command == "process-video":
            return _process(args)
        if args.command == "benchmark":
            return _benchmark(args)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 1
    return parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
