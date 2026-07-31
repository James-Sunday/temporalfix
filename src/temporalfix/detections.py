"""Validated detection interchange data."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ObjectArray = NDArray[np.object_]


class Provenance(StrEnum):
    """Origin of a TemporalFix output row."""

    DIRECT = "direct"
    SMOOTHED = "smoothed"
    RECOVERED = "recovered"
    PREDICTED = "predicted"
    TENTATIVE = "tentative"


def _vector(
    value: ArrayLike | None,
    *,
    length: int,
    dtype: np.dtype[Any],
    default: float | bool,
    name: str,
) -> NDArray[Any]:
    array = (
        np.full(length, default, dtype=dtype)
        if value is None
        else np.asarray(value, dtype=dtype)
    )
    if array.shape != (length,):
        msg = f"{name} must have shape ({length},), got {array.shape}"
        raise ValueError(msg)
    return np.array(array, copy=True)


def _optional_rows(
    value: ArrayLike | None, *, length: int, name: str
) -> NDArray[Any] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != length:
        msg = f"{name} must have first dimension {length}, got {array.shape}"
        raise ValueError(msg)
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


class Detections:
    """A validated, owned collection of XYXY detections.

    Coordinates are continuous half-open ``(x1, y1, x2, y2)`` values. Arrays
    are copied on construction, then made read-only. Metadata mappings are
    shallow-copied; callers should treat nested metadata values as immutable.
    """

    __slots__ = (
        "age",
        "class_id",
        "confidence",
        "is_confirmed",
        "keypoints",
        "masks",
        "metadata",
        "source",
        "time_since_update",
        "track_id",
        "uncertainty",
        "xyxy",
    )

    xyxy: FloatArray
    confidence: FloatArray
    class_id: IntArray
    track_id: IntArray
    masks: NDArray[Any] | None
    keypoints: NDArray[Any] | None
    metadata: tuple[dict[str, Any], ...]
    source: ObjectArray
    uncertainty: FloatArray
    age: IntArray
    time_since_update: IntArray
    is_confirmed: BoolArray

    def __init__(
        self,
        xyxy: ArrayLike,
        confidence: ArrayLike | None = None,
        class_id: ArrayLike | None = None,
        track_id: ArrayLike | None = None,
        masks: ArrayLike | None = None,
        keypoints: ArrayLike | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        source: Sequence[Provenance | str] | None = None,
        uncertainty: ArrayLike | None = None,
        age: ArrayLike | None = None,
        time_since_update: ArrayLike | None = None,
        is_confirmed: ArrayLike | None = None,
    ) -> None:
        """Normalize, validate and freeze fields."""
        boxes = np.asarray(xyxy, dtype=np.float64)
        if boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            msg = f"xyxy must have shape (N, 4), got {boxes.shape}"
            raise ValueError(msg)
        if not np.isfinite(boxes).all():
            raise ValueError("xyxy must contain only finite values")
        if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
            raise ValueError("xyxy requires x2 >= x1 and y2 >= y1")
        self.xyxy = np.array(boxes, copy=True)
        length = len(boxes)
        self.confidence = _vector(
            confidence,
            length=length,
            dtype=np.dtype(np.float64),
            default=1.0,
            name="confidence",
        )
        self.uncertainty = _vector(
            uncertainty,
            length=length,
            dtype=np.dtype(np.float64),
            default=0.0,
            name="uncertainty",
        )
        if not np.isfinite(self.confidence).all() or np.any(
            (self.confidence < 0.0) | (self.confidence > 1.0)
        ):
            raise ValueError("confidence values must be finite and within [0, 1]")
        if not np.isfinite(self.uncertainty).all() or np.any(
            (self.uncertainty < 0.0) | (self.uncertainty > 1.0)
        ):
            raise ValueError("uncertainty values must be finite and within [0, 1]")
        self.class_id = _vector(
            class_id,
            length=length,
            dtype=np.dtype(np.int64),
            default=-1.0,
            name="class_id",
        )
        self.track_id = _vector(
            track_id,
            length=length,
            dtype=np.dtype(np.int64),
            default=-1.0,
            name="track_id",
        )
        self.age = _vector(
            age,
            length=length,
            dtype=np.dtype(np.int64),
            default=0.0,
            name="age",
        )
        self.time_since_update = _vector(
            time_since_update,
            length=length,
            dtype=np.dtype(np.int64),
            default=0.0,
            name="time_since_update",
        )
        self.is_confirmed = _vector(
            is_confirmed,
            length=length,
            dtype=np.dtype(np.bool_),
            default=True,
            name="is_confirmed",
        )
        if np.any(self.age < 0) or np.any(self.time_since_update < 0):
            raise ValueError("age and time_since_update must be non-negative")
        self.source = self._sources(source, length)
        self.metadata = self._metadata(metadata, length)
        self.masks = _optional_rows(masks, length=length, name="masks")
        self.keypoints = _optional_rows(keypoints, length=length, name="keypoints")
        for array in (
            self.xyxy,
            self.confidence,
            self.class_id,
            self.track_id,
            self.source,
            self.uncertainty,
            self.age,
            self.time_since_update,
            self.is_confirmed,
        ):
            array.setflags(write=False)

    @staticmethod
    def _sources(values: Sequence[Provenance | str] | None, length: int) -> ObjectArray:
        if values is None:
            return np.full(length, Provenance.DIRECT, dtype=object)
        if len(values) != length:
            msg = f"source must have length {length}, got {len(values)}"
            raise ValueError(msg)
        try:
            return np.asarray([Provenance(item) for item in values], dtype=object)
        except ValueError as error:
            raise ValueError(f"invalid provenance: {error}") from error

    @staticmethod
    def _metadata(
        values: Sequence[Mapping[str, Any]] | None, length: int
    ) -> tuple[dict[str, Any], ...]:
        if values is None:
            return tuple({} for _ in range(length))
        if len(values) != length:
            msg = f"metadata must have length {length}, got {len(values)}"
            raise ValueError(msg)
        return tuple(dict(item) for item in values)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.xyxy)

    def __repr__(self) -> str:
        """Return a compact informative representation."""
        return (
            f"Detections(n={len(self)}, xyxy_shape={self.xyxy.shape}, "
            f"masks={self.masks is not None}, keypoints={self.keypoints is not None})"
        )

    def __getitem__(self, index: int | slice | BoolArray | IntArray) -> Self:
        """Return a copied row subset without dropping the row dimension."""
        selector: slice | BoolArray | IntArray
        if isinstance(index, int):
            normalized = index if index >= 0 else len(self) + index
            if normalized < 0 or normalized >= len(self):
                raise IndexError("detection index out of range")
            selector = np.asarray([normalized], dtype=np.int64)
        else:
            selector = index
        positions = np.atleast_1d(np.arange(len(self))[selector])
        return type(self)(
            xyxy=self.xyxy[positions],
            confidence=self.confidence[positions],
            class_id=self.class_id[positions],
            track_id=self.track_id[positions],
            masks=None if self.masks is None else self.masks[positions],
            keypoints=None if self.keypoints is None else self.keypoints[positions],
            metadata=[self.metadata[int(position)] for position in positions],
            source=self.source[positions].tolist(),
            uncertainty=self.uncertainty[positions],
            age=self.age[positions],
            time_since_update=self.time_since_update[positions],
            is_confirmed=self.is_confirmed[positions],
        )

    def copy(self) -> Self:
        """Return an independent copy."""
        return self[:]

    @classmethod
    def empty(cls) -> Self:
        """Create a valid empty collection."""
        return cls(xyxy=np.empty((0, 4), dtype=np.float64))

    def to_dict(self) -> dict[str, Any]:
        """Serialize all supported fields to JSON-compatible values."""
        return {
            "schema_version": 1,
            "xyxy": self.xyxy.tolist(),
            "confidence": self.confidence.tolist(),
            "class_id": self.class_id.tolist(),
            "track_id": self.track_id.tolist(),
            "masks": None if self.masks is None else self.masks.tolist(),
            "keypoints": None if self.keypoints is None else self.keypoints.tolist(),
            "metadata": [dict(item) for item in self.metadata],
            "source": [str(item) for item in self.source],
            "uncertainty": self.uncertainty.tolist(),
            "age": self.age.tolist(),
            "time_since_update": self.time_since_update.tolist(),
            "is_confirmed": self.is_confirmed.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize a version-1 dictionary."""
        if data.get("schema_version", 1) != 1:
            raise ValueError("unsupported Detections schema_version")
        allowed = {
            "schema_version",
            "xyxy",
            "confidence",
            "class_id",
            "track_id",
            "masks",
            "keypoints",
            "metadata",
            "source",
            "uncertainty",
            "age",
            "time_since_update",
            "is_confirmed",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown Detections fields: {', '.join(unknown)}")
        if "xyxy" not in data:
            raise ValueError("xyxy is required")
        values = {key: value for key, value in data.items() if key != "schema_version"}
        return cls(**values)
