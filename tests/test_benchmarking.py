from __future__ import annotations

import pytest

from temporalfix.benchmarking import run_synthetic_scaling


def test_synthetic_scaling_records_samples_memory_and_ratios() -> None:
    result = run_synthetic_scaling(
        counts=[1, 3],
        frames=2,
        warmup_frames=1,
        seed=7,
        preset="low_latency",
    )

    assert result["schema_version"] == 1
    assert result["configuration"]["counts"] == [1, 3]
    assert len(result["metrics"]) == 2
    assert all(
        len(metric["latency_ms"]["samples"]) == 2 for metric in result["metrics"]
    )
    assert all(
        metric["python_allocation_peak_bytes"] >= 0 for metric in result["metrics"]
    )
    assert result["scaling"][0]["median_latency_ratio_to_first_count"] == pytest.approx(
        1.0
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"counts": []}, "counts"),
        ({"counts": [0]}, "counts"),
        ({"counts": [1, 1]}, "duplicates"),
        ({"frames": 0}, "frames"),
        ({"warmup_frames": 0}, "warmup"),
    ],
)
def test_synthetic_scaling_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_synthetic_scaling(**kwargs)
