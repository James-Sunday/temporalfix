from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from temporalfix import Detections, Provenance


def test_empty_detections_are_fully_shaped() -> None:
    detections = Detections.empty()
    assert len(detections) == 0
    assert detections.xyxy.shape == (0, 4)
    assert detections.confidence.shape == (0,)
    assert detections.to_dict()["xyxy"] == []


def test_constructor_owns_and_freezes_arrays() -> None:
    boxes = np.asarray([[0.0, 1.0, 2.0, 3.0]])
    detections = Detections(boxes, confidence=[0.8], class_id=[2])
    boxes[0, 0] = 99.0
    assert detections.xyxy[0, 0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        detections.xyxy[0, 0] = 1.0
    assert "Detections(n=1" in repr(detections)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"xyxy": [1, 2, 3, 4]}, "shape"),
        ({"xyxy": [[2, 0, 1, 2]]}, "x2"),
        ({"xyxy": [[0, 2, 1, 1]]}, "y2"),
        ({"xyxy": [[0, 0, np.nan, 1]]}, "finite"),
        ({"xyxy": [[0, 0, np.inf, 1]]}, "finite"),
        ({"xyxy": [[0, 0, 1, 1]], "confidence": [1.2]}, "confidence"),
        ({"xyxy": [[0, 0, 1, 1]], "uncertainty": [-0.1]}, "uncertainty"),
        ({"xyxy": [[0, 0, 1, 1]], "class_id": [1, 2]}, "class_id"),
        ({"xyxy": [[0, 0, 1, 1]], "masks": np.zeros((2, 3, 3))}, "masks"),
    ],
)
def test_invalid_inputs_raise_clear_errors(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Detections(**kwargs)


def test_selection_preserves_rows_and_all_fields() -> None:
    detections = Detections(
        xyxy=[[0, 0, 1, 1], [2, 2, 4, 4]],
        confidence=[0.5, 0.9],
        class_id=[3, 4],
        track_id=[10, 11],
        metadata=[{"name": "a"}, {"name": "b"}],
        source=[Provenance.DIRECT, Provenance.RECOVERED],
    )
    selected = detections[1]
    assert selected.xyxy.shape == (1, 4)
    assert selected.track_id.tolist() == [11]
    assert selected.metadata == ({"name": "b"},)
    assert selected.source.tolist() == [Provenance.RECOVERED]


def test_json_serialization_round_trip_preserves_required_fields() -> None:
    detections = Detections(
        xyxy=[[0, 0, 2, 3]],
        confidence=[0.75],
        class_id=[4],
        track_id=[9],
        masks=np.ones((1, 2, 2), dtype=np.uint8),
        keypoints=[[[1.0, 2.0, 0.9]]],
        metadata=[{"camera": "north"}],
        source=[Provenance.SMOOTHED],
        uncertainty=[0.2],
        age=[5],
        time_since_update=[0],
        is_confirmed=[True],
    )
    payload = json.loads(json.dumps(detections.to_dict()))
    restored = Detections.from_dict(payload)
    np.testing.assert_array_equal(restored.xyxy, detections.xyxy)
    np.testing.assert_array_equal(restored.masks, detections.masks)
    assert restored.to_dict() == detections.to_dict()


def test_unknown_serialized_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        Detections.from_dict({"xyxy": [], "surprise": True})
    with pytest.raises(ValueError, match="schema_version"):
        Detections.from_dict({"schema_version": 2, "xyxy": []})


@given(
    x1=arrays(np.float64, 20, elements=st.floats(-1e6, 1e6, allow_nan=False)),
    width=arrays(np.float64, 20, elements=st.floats(0, 1e4, allow_nan=False)),
    y1=arrays(np.float64, 20, elements=st.floats(-1e6, 1e6, allow_nan=False)),
    height=arrays(np.float64, 20, elements=st.floats(0, 1e4, allow_nan=False)),
    confidence=arrays(np.float64, 20, elements=st.floats(0, 1, allow_nan=False)),
)
def test_property_lengths_geometry_and_serialization(
    x1: np.ndarray,
    width: np.ndarray,
    y1: np.ndarray,
    height: np.ndarray,
    confidence: np.ndarray,
) -> None:
    boxes = np.column_stack((x1, y1, x1 + width, y1 + height))
    detections = Detections(boxes, confidence=confidence)
    assert np.all(detections.xyxy[:, 2:] >= detections.xyxy[:, :2])
    assert len(detections.confidence) == len(detections)
    restored = Detections.from_dict(detections.to_dict())
    np.testing.assert_array_equal(restored.xyxy, detections.xyxy)
