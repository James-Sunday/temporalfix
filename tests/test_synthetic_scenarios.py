from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from temporalfix import Detections, TemporalFixConfig, TemporalRepairer


def _frame(x: float, *, confidence: float = 0.9, class_id: int = 1) -> Detections:
    return Detections(
        [[x, 0.0, x + 8.0, 8.0]],
        confidence=[confidence],
        class_id=[class_id],
    )


def _config(**overrides: object) -> TemporalFixConfig:
    return TemporalFixConfig.from_dict(
        {
            "min_confirmed_observations": 1,
            "box_smoothing": "none",
            "min_iou": 0.0,
            **overrides,
        }
    )


def test_accelerating_object_keeps_one_identity() -> None:
    repairer = TemporalRepairer(_config())
    ids = [
        repairer.update(_frame(position)).track_id[0]
        for position in (0.0, 1.0, 3.0, 6.0, 10.0)
    ]
    assert ids == [1, 1, 1, 1, 1]


def test_stationary_jitter_is_reduced_by_ema() -> None:
    raw = np.asarray([0.0, 1.0, -1.0, 0.8, -0.7, 0.2])
    repairer = TemporalRepairer(_config(box_smoothing="ema", ema_alpha=0.4))
    fixed = np.asarray(
        [repairer.update(_frame(float(value))).xyxy[0, 0] for value in raw]
    )
    assert np.var(fixed) < np.var(raw)


def test_long_occlusion_expires_identity_and_reentry_is_new() -> None:
    repairer = TemporalRepairer(_config(max_missing_frames=2))
    assert repairer.update(_frame(0)).track_id.tolist() == [1]
    repairer.update(Detections.empty())
    repairer.update(Detections.empty())
    assert len(repairer.update(Detections.empty())) == 0
    assert repairer.update(_frame(0)).track_id.tolist() == [2]


def test_confidence_collapse_is_smoothed_but_not_increased_without_evidence() -> None:
    repairer = TemporalRepairer(
        _config(confidence_alpha=0.5, confidence_stabilization=True)
    )
    high = repairer.update(_frame(0, confidence=1.0)).confidence[0]
    collapsed = repairer.update(_frame(0, confidence=0.1)).confidence[0]
    missed = repairer.update(Detections.empty()).confidence[0]
    assert high > collapsed > 0.1
    assert missed <= collapsed


def test_new_object_entering_during_gap_gets_distinct_identity() -> None:
    repairer = TemporalRepairer(_config(max_missing_frames=3, min_iou=0.1))
    assert repairer.update(_frame(0)).track_id.tolist() == [1]
    output = repairer.update(_frame(100, class_id=2))
    assert output.track_id.tolist() == [1, 2]
    assert output.time_since_update.tolist() == [1, 0]


def test_regression_fixture_preserves_lifecycle_and_serialization() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "temporal_sequence.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    repairer = TemporalRepairer(TemporalFixConfig.from_dict(fixture["config"]))
    observed = []
    for frame in fixture["frames"]:
        output = repairer.update(Detections.from_dict(frame))
        observed.append(
            {
                "xyxy": output.xyxy.tolist(),
                "track_id": output.track_id.tolist(),
                "class_id": output.class_id.tolist(),
                "source": [str(item) for item in output.source],
                "time_since_update": output.time_since_update.tolist(),
            }
        )
    assert observed == fixture["expected"]
