from __future__ import annotations

import numpy as np

from temporalfix import Detections, Provenance
from temporalfix.adapters.numpy import from_numpy, to_numpy


def test_numpy_adapter_handles_empty_detections() -> None:
    detections = from_numpy(np.empty((0, 4)))
    converted = to_numpy(detections)
    assert converted["xyxy"].shape == (0, 4)  # type: ignore[union-attr]


def test_numpy_adapter_preserves_every_field() -> None:
    detections = Detections(
        [[0, 0, 1, 1]],
        confidence=[0.8],
        class_id=[2],
        track_id=[4],
        metadata=[{"x": 1}],
        source=[Provenance.RECOVERED],
        uncertainty=[0.4],
        age=[3],
        time_since_update=[1],
        is_confirmed=[True],
    )
    converted = to_numpy(detections)
    assert converted["metadata"] == ({"x": 1},)
    assert converted["source"].tolist() == [Provenance.RECOVERED]  # type: ignore[union-attr]
    assert converted["uncertainty"].tolist() == [0.4]  # type: ignore[union-attr]
    assert converted["age"].tolist() == [3]  # type: ignore[union-attr]
