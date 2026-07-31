from __future__ import annotations

import numpy as np
import pytest

from temporalfix import (
    BoxSmoothing,
    ClassVoting,
    Detections,
    Provenance,
    TemporalFixConfig,
    TemporalRepairer,
)


def detection(x: float = 0.0, confidence: float = 0.9, class_id: int = 1) -> Detections:
    return Detections(
        [[x, 0.0, x + 10.0, 10.0]],
        confidence=[confidence],
        class_id=[class_id],
    )


def immediate_config(**values: object) -> TemporalFixConfig:
    return TemporalFixConfig.from_dict(
        {
            "min_confirmed_observations": 1,
            "box_smoothing": "none",
            **values,
        }
    )


def test_confirmation_suppresses_one_frame_false_positive() -> None:
    repairer = TemporalRepairer(
        TemporalFixConfig(
            min_confirmed_observations=2,
            max_tentative_age=2,
            max_missing_frames=0,
        )
    )
    assert len(repairer.update(detection())) == 0
    assert len(repairer.update(Detections.empty())) == 0


def test_track_creation_confirmation_and_unique_ids() -> None:
    repairer = TemporalRepairer(
        TemporalFixConfig(
            min_confirmed_observations=2,
            box_smoothing=BoxSmoothing.NONE,
        )
    )
    assert len(repairer.update(detection())) == 0
    output = repairer.update(detection(0.2))
    assert output.track_id.tolist() == [1]
    assert output.is_confirmed.tolist() == [True]
    assert output.source.tolist() == [Provenance.DIRECT]
    assert len(set(output.track_id.tolist())) == len(output)


def test_gap_recovery_decays_confidence_and_increases_uncertainty() -> None:
    repairer = TemporalRepairer(
        immediate_config(
            max_missing_frames=2,
            confidence_decay=0.5,
            uncertainty_growth=0.2,
        )
    )
    direct = repairer.update(detection(confidence=0.8))
    recovered = repairer.update(Detections.empty())
    predicted = repairer.update(Detections.empty())
    expired = repairer.update(Detections.empty())
    assert recovered.source.tolist() == [Provenance.RECOVERED]
    assert predicted.source.tolist() == [Provenance.PREDICTED]
    assert recovered.confidence[0] <= direct.confidence[0]
    assert predicted.confidence[0] <= recovered.confidence[0]
    assert recovered.uncertainty[0] > direct.uncertainty[0]
    assert predicted.uncertainty[0] > recovered.uncertainty[0]
    assert len(expired) == 0


def test_no_confidence_increase_without_observation() -> None:
    repairer = TemporalRepairer(immediate_config(max_missing_frames=4))
    values = [repairer.update(detection(confidence=0.9)).confidence[0]]
    values.extend(repairer.update(Detections.empty()).confidence[0] for _ in range(4))
    assert values == sorted(values, reverse=True)


def test_ema_smoothing_reduces_single_frame_jump() -> None:
    repairer = TemporalRepairer(
        immediate_config(box_smoothing="ema", ema_alpha=0.5, min_iou=0.0)
    )
    repairer.update(detection(0.0))
    output = repairer.update(detection(10.0))
    assert 0.0 < output.xyxy[0, 0] < 10.0
    assert output.source.tolist() == [Provenance.SMOOTHED]


def test_kalman_predicts_constant_velocity_gap() -> None:
    repairer = TemporalRepairer(immediate_config(box_smoothing="kalman", min_iou=0.0))
    repairer.update(detection(0.0))
    observed = repairer.update(detection(2.0))
    predicted = repairer.update(Detections.empty())
    assert predicted.xyxy[0, 0] >= observed.xyxy[0, 0]


def test_temporary_class_switch_is_stabilized_and_diagnosed() -> None:
    repairer = TemporalRepairer(
        immediate_config(
            class_voting=ClassVoting.MAJORITY,
            class_switch_threshold=1.0,
            class_gating=False,
            min_iou=0.0,
        )
    )
    repairer.update(detection(class_id=1))
    output = repairer.update(detection(class_id=2))
    assert output.class_id.tolist() == [1]
    assert output.metadata[0]["class_switch"] is False
    switched = repairer.update(detection(class_id=2))
    assert switched.class_id.tolist() == [2]
    assert switched.metadata[0]["class_switch"] is True
    assert switched.metadata[0]["previous_class_id"] == 1


def test_streams_and_reset_do_not_leak_state() -> None:
    repairer = TemporalRepairer(immediate_config())
    first_a = repairer.update(detection(), stream_id="a")
    first_b = repairer.update(detection(100), stream_id="b")
    assert first_a.track_id.tolist() == [1]
    assert first_b.track_id.tolist() == [1]
    repairer.reset(stream_id="a")
    reset_a = repairer.update(detection(), stream_id="a")
    continued_b = repairer.update(detection(100), stream_id="b")
    assert reset_a.track_id.tolist() == [1]
    assert continued_b.age.tolist() == [2]
    repairer.reset()
    assert repairer.update(detection(), stream_id="b").age.tolist() == [1]


def test_timestamp_policy_allows_repeat_and_rejects_backwards() -> None:
    repairer = TemporalRepairer(immediate_config())
    repairer.update(detection(), timestamp=10.0)
    repairer.update(detection(), timestamp=10.0)
    with pytest.raises(ValueError, match="backwards"):
        repairer.update(detection(), timestamp=9.0)
    with pytest.raises(ValueError, match="finite"):
        TemporalRepairer(immediate_config()).update(detection(), timestamp=np.nan)


def test_overlapping_different_classes_can_be_gated() -> None:
    repairer = TemporalRepairer(
        immediate_config(class_gating=True, output_tentative=True)
    )
    first = repairer.update(detection(class_id=1))
    second = repairer.update(detection(class_id=2))
    assert first.track_id.tolist() == [1]
    assert second.track_id.tolist() == [1, 2]


def test_deterministic_crossing_scenario() -> None:
    config = immediate_config(min_iou=0.0)
    sequences = []
    for _ in range(2):
        repairer = TemporalRepairer(config)
        run = []
        for left, right in ((0, 20), (5, 15), (9, 11), (14, 6)):
            output = repairer.update(
                Detections(
                    [[left, 0, left + 4, 4], [right, 0, right + 4, 4]],
                    confidence=[0.9, 0.9],
                    class_id=[1, 1],
                )
            )
            run.append((output.track_id.tolist(), output.xyxy.tolist()))
        sequences.append(run)
    assert sequences[0] == sequences[1]
