from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

import temporalfix.adapters.ultralytics as ultralytics_adapter
from temporalfix import Detections, Provenance
from temporalfix.adapters.ultralytics import (
    LossyUltralyticsConversionError,
    from_ultralytics,
    to_ultralytics,
)


class FakePayload:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class FakeBoxes:
    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data)
        self.xyxy = self.data[:, :4]
        self.conf = self.data[:, -2]
        self.cls = self.data[:, -1]
        self.id = self.data[:, -3] if self.data.shape[1] == 7 else None


class FakeResults:
    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        **_: object,
    ) -> None:
        self.orig_img = orig_img
        self.path = path
        self.names = names
        self.boxes = None if boxes is None else FakeBoxes(boxes)
        self.masks = None if masks is None else FakePayload(masks)
        self.keypoints = None if keypoints is None else FakePayload(keypoints)


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("ultralytics")
    engine = ModuleType("ultralytics.engine")
    results = ModuleType("ultralytics.engine.results")
    results.Results = FakeResults
    monkeypatch.setitem(sys.modules, "ultralytics", package)
    monkeypatch.setitem(sys.modules, "ultralytics.engine", engine)
    monkeypatch.setitem(sys.modules, "ultralytics.engine.results", results)


def test_from_ultralytics_preserves_public_detection_fields(
    fake_ultralytics: None,
) -> None:
    assert fake_ultralytics is None
    result = FakeResults(
        orig_img=np.zeros((20, 30, 3), dtype=np.uint8),
        path="frame.jpg",
        names={2: "object"},
        boxes=np.array([[1, 2, 11, 12, 42, 0.8, 2]], dtype=np.float32),
        masks=np.ones((1, 20, 30), dtype=np.uint8),
        keypoints=np.array([[[3, 4, 0.9]]], dtype=np.float32),
    )

    converted = from_ultralytics(result)

    np.testing.assert_allclose(converted.xyxy, [[1, 2, 11, 12]])
    np.testing.assert_allclose(converted.confidence, [0.8])
    np.testing.assert_array_equal(converted.class_id, [2])
    np.testing.assert_array_equal(converted.track_id, [42])
    np.testing.assert_array_equal(converted.masks, result.masks.data)
    np.testing.assert_allclose(converted.keypoints, result.keypoints.data)


def test_from_ultralytics_handles_no_boxes(fake_ultralytics: None) -> None:
    assert fake_ultralytics is None
    result = FakeResults(
        orig_img=np.zeros((5, 5, 3), dtype=np.uint8),
        path="",
        names={},
    )
    assert len(from_ultralytics(result)) == 0


def test_to_ultralytics_preserves_representable_fields(
    fake_ultralytics: None,
) -> None:
    assert fake_ultralytics is None
    detections = Detections(
        xyxy=[[1, 2, 5, 7]],
        confidence=[0.75],
        class_id=[3],
        track_id=[8],
        masks=np.ones((1, 4, 4), dtype=np.uint8),
        keypoints=[[[2, 3, 0.5]]],
    )

    result = to_ultralytics(
        detections,
        orig_img=np.zeros((10, 10, 3), dtype=np.uint8),
        names={3: "item"},
    )

    assert isinstance(result, FakeResults)
    np.testing.assert_allclose(result.boxes.data, [[1, 2, 5, 7, 8, 0.75, 3]])
    np.testing.assert_array_equal(result.masks.data, detections.masks)
    np.testing.assert_array_equal(result.keypoints.data, detections.keypoints)


def test_to_ultralytics_requires_explicit_loss_opt_in(
    fake_ultralytics: None,
) -> None:
    assert fake_ultralytics is None
    detections = Detections(
        xyxy=[[0, 0, 2, 2]],
        source=[Provenance.PREDICTED],
        uncertainty=[0.4],
        age=[2],
    )

    with pytest.raises(
        LossyUltralyticsConversionError,
        match="source, uncertainty, age",
    ):
        to_ultralytics(
            detections,
            orig_img=np.zeros((3, 3, 3), dtype=np.uint8),
        )

    result = to_ultralytics(
        detections,
        orig_img=np.zeros((3, 3, 3), dtype=np.uint8),
        allow_lossy=True,
    )
    assert isinstance(result, FakeResults)


def test_ultralytics_missing_extra_has_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> ModuleType:
        raise ModuleNotFoundError(name="ultralytics")

    monkeypatch.setattr(ultralytics_adapter.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"temporalfix\[ultralytics\]"):
        from_ultralytics(object())


@pytest.mark.integration
def test_installed_ultralytics_public_api_round_trip() -> None:
    results_module = pytest.importorskip(
        "ultralytics.engine.results",
        reason="optional Ultralytics extra not installed",
    )

    result = results_module.Results(
        orig_img=np.zeros((8, 8, 3), dtype=np.uint8),
        path="frame.jpg",
        names={0: "object"},
        boxes=np.array([[1, 1, 5, 5, 0.9, 0]], dtype=np.float32),
    )
    converted = from_ultralytics(result)
    rebuilt = to_ultralytics(converted, orig_img=result.orig_img, names=result.names)
    np.testing.assert_allclose(rebuilt.boxes.xyxy, result.boxes.xyxy)
