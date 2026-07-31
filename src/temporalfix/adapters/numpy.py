"""Lossless NumPy adapter helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from temporalfix.detections import Detections


def from_numpy(
    xyxy: ArrayLike,
    confidence: ArrayLike | None = None,
    class_id: ArrayLike | None = None,
    track_id: ArrayLike | None = None,
    **fields: Any,
) -> Detections:
    """Construct :class:`Detections` from NumPy-compatible arrays."""
    return Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        track_id=track_id,
        **fields,
    )


def to_numpy(
    detections: Detections, *, copy: bool = True
) -> dict[str, NDArray[Any] | tuple[dict[str, Any], ...] | None]:
    """Return every field without silently dropping temporal information."""

    def maybe_copy(array: NDArray[Any] | None) -> NDArray[Any] | None:
        if array is None:
            return None
        return np.array(array, copy=True) if copy else array

    return {
        "xyxy": maybe_copy(detections.xyxy),
        "confidence": maybe_copy(detections.confidence),
        "class_id": maybe_copy(detections.class_id),
        "track_id": maybe_copy(detections.track_id),
        "masks": maybe_copy(detections.masks),
        "keypoints": maybe_copy(detections.keypoints),
        "metadata": tuple(dict(item) for item in detections.metadata),
        "source": maybe_copy(detections.source),
        "uncertainty": maybe_copy(detections.uncertainty),
        "age": maybe_copy(detections.age),
        "time_since_update": maybe_copy(detections.time_since_update),
        "is_confirmed": maybe_copy(detections.is_confirmed),
    }
