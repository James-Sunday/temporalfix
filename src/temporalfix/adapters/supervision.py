"""Optional loss-aware Supervision ``Detections`` conversion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from temporalfix.detections import Detections

_INSTALL_GUIDANCE = (
    "Supervision support requires the optional dependency. "
    'Install it with `pip install "temporalfix[supervision]"`.'
)
_PREFIX = "temporalfix."
_SOURCE = f"{_PREFIX}source"
_UNCERTAINTY = f"{_PREFIX}uncertainty"
_AGE = f"{_PREFIX}age"
_TIME_SINCE_UPDATE = f"{_PREFIX}time_since_update"
_IS_CONFIRMED = f"{_PREFIX}is_confirmed"
_KEYPOINTS = f"{_PREFIX}keypoints"
_METADATA = f"{_PREFIX}metadata"
_RESERVED = {
    _SOURCE,
    _UNCERTAINTY,
    _AGE,
    _TIME_SINCE_UPDATE,
    _IS_CONFIRMED,
    _KEYPOINTS,
    _METADATA,
}


def _detections_type() -> Any:
    try:
        module = importlib.import_module("supervision")
    except ModuleNotFoundError as error:
        if error.name == "supervision" or (
            error.name is not None and error.name.startswith("supervision.")
        ):
            raise ImportError(_INSTALL_GUIDANCE) from error
        raise
    return module.Detections


def _row_value(value: Any, index: int) -> Any:
    item = value[index]
    return np.array(item, copy=True) if isinstance(item, np.ndarray) else item


def _metadata_from_data(
    data: Mapping[str, Any], length: int
) -> tuple[dict[str, Any], ...] | None:
    stored = data.get(_METADATA)
    if stored is None:
        metadata: list[dict[str, Any]] = [{} for _ in range(length)]
    else:
        if not isinstance(stored, Sequence) or len(stored) != length:
            raise ValueError(f"{_METADATA} must contain one mapping per detection")
        metadata = []
        for item in stored:
            if not isinstance(item, Mapping):
                raise TypeError(f"{_METADATA} entries must be mappings")
            metadata.append(dict(item))

    external = {key: value for key, value in data.items() if key not in _RESERVED}
    for index in range(length):
        if external:
            metadata[index]["supervision_data"] = {
                key: _row_value(value, index) for key, value in external.items()
            }
    return tuple(metadata) if any(metadata) else None


def from_supervision(value: object) -> Detections:
    """Convert Supervision detections and preserve namespaced temporal fields."""
    detections_type = _detections_type()
    if not isinstance(value, detections_type):
        raise TypeError("value must be a supervision.Detections object")

    data = getattr(value, "data", {})
    if not isinstance(data, Mapping):
        raise TypeError("supervision Detections.data must be a mapping")
    length = len(value.xyxy)
    return Detections(
        xyxy=value.xyxy,
        confidence=value.confidence,
        class_id=value.class_id,
        track_id=value.tracker_id,
        masks=value.mask,
        keypoints=data.get(_KEYPOINTS),
        metadata=_metadata_from_data(data, length),
        source=data.get(_SOURCE),
        uncertainty=data.get(_UNCERTAINTY),
        age=data.get(_AGE),
        time_since_update=data.get(_TIME_SINCE_UPDATE),
        is_confirmed=data.get(_IS_CONFIRMED),
    )


def _external_data(
    metadata: tuple[dict[str, Any], ...],
) -> dict[str, list[Any]]:
    keys: set[str] = set()
    rows: list[Mapping[str, Any]] = []
    for item in metadata:
        candidate = item.get("supervision_data", {})
        if not isinstance(candidate, Mapping):
            raise TypeError("metadata supervision_data must be a mapping")
        rows.append(candidate)
        keys.update(str(key) for key in candidate)
    return {key: [row.get(key) for row in rows] for key in sorted(keys)}


def to_supervision(detections: Detections) -> Any:
    """Create Supervision detections without dropping TemporalFix state."""
    data: dict[str, Any] = _external_data(detections.metadata)
    data.update(
        {
            _SOURCE: np.asarray([str(item) for item in detections.source]),
            _UNCERTAINTY: np.array(detections.uncertainty, copy=True),
            _AGE: np.array(detections.age, copy=True),
            _TIME_SINCE_UPDATE: np.array(detections.time_since_update, copy=True),
            _IS_CONFIRMED: np.array(detections.is_confirmed, copy=True),
            _METADATA: [dict(item) for item in detections.metadata],
        }
    )
    if detections.keypoints is not None:
        data[_KEYPOINTS] = np.array(detections.keypoints, copy=True)

    detections_type = _detections_type()
    return detections_type(
        xyxy=np.array(detections.xyxy, copy=True),
        mask=(
            None if detections.masks is None else np.array(detections.masks, copy=True)
        ),
        confidence=np.array(detections.confidence, copy=True),
        class_id=np.array(detections.class_id, copy=True),
        tracker_id=np.array(detections.track_id, copy=True),
        data=data,
    )
