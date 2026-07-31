"""Optional conversion to and from public Ultralytics ``Results`` fields."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from temporalfix.detections import Detections, Provenance

_INSTALL_GUIDANCE = (
    "Ultralytics support requires the optional dependency. "
    'Install it with `pip install "temporalfix[ultralytics]"`. '
    "Ultralytics is distributed under AGPL-3.0 or a commercial licence; "
    "review its licence before use."
)
_MAX_EXACT_FLOAT_INTEGER = 2**53


class LossyUltralyticsConversionError(ValueError):
    """Raised when ``Results`` cannot represent TemporalFix-only fields."""


def _results_type() -> Any:
    try:
        module = importlib.import_module("ultralytics.engine.results")
    except ModuleNotFoundError as error:
        if error.name == "ultralytics" or (
            error.name is not None and error.name.startswith("ultralytics.")
        ):
            raise ImportError(_INSTALL_GUIDANCE) from error
        raise
    return module.Results


def _array(value: Any, *, dtype: np.dtype[Any] | None = None) -> NDArray[Any]:
    """Copy a public tensor/array field onto CPU as a NumPy array."""
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    return np.array(current, dtype=dtype, copy=True)


def _integer_ids(value: Any, *, name: str) -> NDArray[np.int64]:
    array = _array(value, dtype=np.dtype(np.float64))
    if not np.isfinite(array).all() or not np.equal(array, np.trunc(array)).all():
        raise ValueError(f"Ultralytics {name} values must be finite integers")
    return array.astype(np.int64)


def _payload_data(value: Any) -> NDArray[Any] | None:
    if value is None:
        return None
    data = getattr(value, "data", None)
    return None if data is None else _array(data)


def from_ultralytics(result: object) -> Detections:
    """Convert one Ultralytics ``Results`` object without importing it eagerly.

    Bounding boxes, confidence, class IDs, tracking IDs, masks and keypoints are
    copied. A result with no boxes becomes a valid empty :class:`Detections`.
    """
    results_type = _results_type()
    if not isinstance(result, results_type):
        raise TypeError("result must be an ultralytics.engine.results.Results object")

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return Detections.empty()

    return Detections(
        xyxy=_array(boxes.xyxy, dtype=np.dtype(np.float64)),
        confidence=_array(boxes.conf, dtype=np.dtype(np.float64)),
        class_id=_integer_ids(boxes.cls, name="class IDs"),
        track_id=(
            None
            if getattr(boxes, "id", None) is None
            else _integer_ids(boxes.id, name="tracking IDs")
        ),
        masks=_payload_data(getattr(result, "masks", None)),
        keypoints=_payload_data(getattr(result, "keypoints", None)),
    )


def _lossy_fields(detections: Detections) -> list[str]:
    fields: list[str] = []
    if any(detections.metadata):
        fields.append("metadata")
    if any(source != Provenance.DIRECT for source in detections.source):
        fields.append("source")
    if np.any(detections.uncertainty != 0.0):
        fields.append("uncertainty")
    if np.any(detections.age != 0):
        fields.append("age")
    if np.any(detections.time_since_update != 0):
        fields.append("time_since_update")
    if np.any(~detections.is_confirmed):
        fields.append("is_confirmed")
    return fields


def _default_names(class_id: NDArray[np.int64]) -> dict[int, str]:
    return {
        int(identifier): str(int(identifier))
        for identifier in np.unique(class_id)
        if identifier >= 0
    }


def to_ultralytics(
    detections: Detections,
    *,
    orig_img: ArrayLike,
    path: str = "",
    names: Mapping[int, str] | None = None,
    allow_lossy: bool = False,
) -> Any:
    """Create an Ultralytics ``Results`` object.

    Ultralytics has no documented fields for TemporalFix provenance,
    uncertainty, lifecycle state or row metadata. By default this function
    raises rather than silently discard non-default values. Set
    ``allow_lossy=True`` only when that loss is acceptable.
    """
    lossy = _lossy_fields(detections)
    if lossy and not allow_lossy:
        joined = ", ".join(lossy)
        raise LossyUltralyticsConversionError(
            "Ultralytics Results cannot represent these non-default fields: "
            f"{joined}. Pass allow_lossy=True to discard them explicitly."
        )
    if np.any(np.abs(detections.track_id) > _MAX_EXACT_FLOAT_INTEGER):
        raise ValueError("track IDs exceed Ultralytics' exactly representable range")

    image = np.asarray(orig_img)
    if image.ndim < 2:
        raise ValueError("orig_img must have at least height and width dimensions")

    tracked = bool(np.any(detections.track_id >= 0))
    columns: list[NDArray[Any]] = [detections.xyxy]
    if tracked:
        columns.append(detections.track_id.reshape(-1, 1))
    columns.extend(
        [
            detections.confidence.reshape(-1, 1),
            detections.class_id.reshape(-1, 1),
        ]
    )
    boxes = np.column_stack(columns).astype(np.float64, copy=False)
    results_type = _results_type()
    resolved_names = (
        _default_names(detections.class_id)
        if names is None
        else {int(key): str(value) for key, value in names.items()}
    )
    return results_type(
        orig_img=image,
        path=path,
        names=resolved_names,
        boxes=boxes,
        masks=detections.masks,
        keypoints=detections.keypoints,
    )
