"""Measured synthetic scaling benchmark utilities."""

from __future__ import annotations

import platform
import time
import tracemalloc
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from temporalfix._version import __version__
from temporalfix.config import TemporalFixConfig
from temporalfix.detections import Detections
from temporalfix.repairer import TemporalRepairer


def _frame(count: int, generator: np.random.Generator, frame_index: int) -> Detections:
    base = np.arange(count, dtype=np.float64) * 20.0
    motion = frame_index * 0.1
    jitter = generator.normal(0.0, 0.05, size=count)
    x1 = base + motion + jitter
    boxes = np.column_stack((x1, np.zeros(count), x1 + 10.0, np.full(count, 10.0)))
    return Detections(
        xyxy=boxes,
        confidence=np.full(count, 0.9),
        class_id=np.zeros(count, dtype=np.int64),
    )


def _measure_count(
    count: int,
    *,
    frames: int,
    warmup_frames: int,
    seed: int,
    config: TemporalFixConfig,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed + count)
    repairer = TemporalRepairer(config)
    frame_index = 0
    for _ in range(warmup_frames):
        repairer.update(_frame(count, generator, frame_index))
        frame_index += 1

    samples_ns: list[int] = []
    output_rows = 0
    for _ in range(frames):
        detections = _frame(count, generator, frame_index)
        frame_index += 1
        started = time.perf_counter_ns()
        output_rows = len(repairer.update(detections))
        samples_ns.append(time.perf_counter_ns() - started)

    memory_frame = _frame(count, generator, frame_index)
    tracemalloc.start()
    try:
        repairer.update(memory_frame)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    samples_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    return {
        "detections_per_frame": count,
        "output_rows": output_rows,
        "latency_ms": {
            "median": float(np.median(samples_ms)),
            "p95": float(np.percentile(samples_ms, 95)),
            "samples": samples_ms.tolist(),
        },
        "python_allocation_peak_bytes": int(peak_bytes),
    }


def run_synthetic_scaling(
    *,
    counts: Sequence[int] = (10, 50, 100, 500),
    frames: int = 20,
    warmup_frames: int = 3,
    seed: int = 0,
    preset: str = "balanced",
) -> dict[str, Any]:
    """Measure update latency and traced peak allocation at several row counts."""
    normalized_counts = tuple(int(count) for count in counts)
    if not normalized_counts or any(count < 1 for count in normalized_counts):
        raise ValueError("counts must contain positive integers")
    if len(set(normalized_counts)) != len(normalized_counts):
        raise ValueError("counts must not contain duplicates")
    if frames < 1:
        raise ValueError("frames must be positive")
    if warmup_frames < 1:
        raise ValueError("warmup_frames must be at least one")

    config = TemporalFixConfig.preset(preset)
    metrics = [
        _measure_count(
            count,
            frames=frames,
            warmup_frames=warmup_frames,
            seed=seed,
            config=config,
        )
        for count in normalized_counts
    ]
    baseline = float(metrics[0]["latency_ms"]["median"])
    scaling = [
        {
            "detections_per_frame": metric["detections_per_frame"],
            "median_latency_ratio_to_first_count": (
                None
                if baseline == 0.0
                else float(metric["latency_ms"]["median"]) / baseline
            ),
        }
        for metric in metrics
    ]
    return {
        "schema_version": 1,
        "benchmark": "temporalfix.synthetic_scaling",
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "counts": list(normalized_counts),
            "frames": frames,
            "warmup_frames": warmup_frames,
            "seed": seed,
            "preset": preset,
            "repair_config": config.to_dict(),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "temporalfix": __version__,
        },
        "metrics": metrics,
        "scaling": scaling,
        "limitations": [
            "Synthetic boxes do not measure detector or task accuracy.",
            "Timing covers TemporalRepairer.update only, not input construction.",
            "Peak memory uses tracemalloc for one separate update and is not "
            "process RSS or GPU memory.",
            "Results describe only the recorded environment and configuration.",
        ],
    }
