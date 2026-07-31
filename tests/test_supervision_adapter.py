from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import temporalfix.adapters.supervision as supervision_adapter
from temporalfix import Detections, Provenance
from temporalfix.adapters.supervision import from_supervision, to_supervision


class FakeSupervisionDetections:
    def __init__(
        self,
        xyxy: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        class_id: np.ndarray | None = None,
        tracker_id: np.ndarray | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.xyxy = (
            np.empty((0, 4), dtype=np.float64) if xyxy is None else np.asarray(xyxy)
        )
        self.mask = mask
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.data = {} if data is None else data


@pytest.fixture
def fake_supervision(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("supervision")
    module.Detections = FakeSupervisionDetections
    monkeypatch.setitem(sys.modules, "supervision", module)


def test_supervision_round_trip_preserves_all_temporal_fields(
    fake_supervision: None,
) -> None:
    assert fake_supervision is None
    original = Detections(
        xyxy=[[1, 2, 6, 8]],
        confidence=[0.7],
        class_id=[4],
        track_id=[9],
        masks=np.ones((1, 3, 3), dtype=np.uint8),
        keypoints=[[[2, 3, 0.8]]],
        metadata=[{"event": "switch"}],
        source=[Provenance.RECOVERED],
        uncertainty=[0.3],
        age=[5],
        time_since_update=[1],
        is_confirmed=[False],
    )

    supervision_value = to_supervision(original)
    restored = from_supervision(supervision_value)

    assert restored.to_dict() == original.to_dict()


def test_from_supervision_preserves_external_data_as_metadata(
    fake_supervision: None,
) -> None:
    assert fake_supervision is None
    value = FakeSupervisionDetections(
        xyxy=np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32),
        confidence=np.array([0.8, 0.9]),
        class_id=np.array([1, 2]),
        data={"class_name": ["cat", "dog"]},
    )

    converted = from_supervision(value)

    assert converted.metadata[0]["supervision_data"] == {"class_name": "cat"}
    assert converted.metadata[1]["supervision_data"] == {"class_name": "dog"}


def test_supervision_empty_conversion(fake_supervision: None) -> None:
    assert fake_supervision is None
    converted = from_supervision(FakeSupervisionDetections())
    assert len(converted) == 0
    assert len(to_supervision(converted).xyxy) == 0


def test_supervision_missing_extra_has_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> ModuleType:
        raise ModuleNotFoundError(name="supervision")

    monkeypatch.setattr(supervision_adapter.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"temporalfix\[supervision\]"):
        from_supervision(object())


@pytest.mark.integration
def test_installed_supervision_public_api_round_trip() -> None:
    supervision = pytest.importorskip(
        "supervision", reason="optional Supervision extra not installed"
    )
    value = supervision.Detections(
        xyxy=np.array([[1, 1, 4, 4]], dtype=np.float32),
        confidence=np.array([0.8], dtype=np.float32),
        class_id=np.array([2]),
        tracker_id=np.array([7]),
    )
    converted = from_supervision(value)
    restored = to_supervision(converted)
    np.testing.assert_allclose(restored.xyxy, value.xyxy)
    np.testing.assert_array_equal(restored.tracker_id, value.tracker_id)
