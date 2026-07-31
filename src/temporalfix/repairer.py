"""Stateful temporal detection repair."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from temporalfix.association import associate
from temporalfix.config import BoxSmoothing, ClassVoting, TemporalFixConfig
from temporalfix.detections import Detections, Provenance
from temporalfix.filters import BoxFilter, KalmanBoxFilter

Filter = BoxFilter | KalmanBoxFilter


def _make_filter(box: NDArray[np.float64], config: TemporalFixConfig) -> Filter:
    if config.box_smoothing == BoxSmoothing.KALMAN:
        return KalmanBoxFilter(
            measurement=box,
            process_noise=config.kalman_process_noise,
            measurement_noise=config.kalman_measurement_noise,
        )
    alpha = config.ema_alpha if config.box_smoothing == BoxSmoothing.EMA else None
    return BoxFilter(value=box.copy(), alpha=alpha)


@dataclass(slots=True)
class _Track:
    track_id: int
    box_filter: Filter
    box: NDArray[np.float64]
    confidence: float
    class_id: int
    config: TemporalFixConfig
    observations: int = 1
    age: int = 1
    missing: int = 0
    uncertainty: float = 0.1
    source: Provenance = Provenance.TENTATIVE
    confirmed: bool = False
    class_history: deque[tuple[int, float]] = field(default_factory=deque)
    class_evidence: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    mask: NDArray[Any] | None = None
    keypoints: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        self.class_history = deque(maxlen=self.config.class_history_size)
        self.class_history.append((self.class_id, self.confidence))
        self.class_evidence[self.class_id] = self.confidence
        self.confirmed = self.observations >= self.config.min_confirmed_observations
        self.source = Provenance.DIRECT if self.confirmed else Provenance.TENTATIVE

    def prepare(self, delta: float) -> None:
        """Predict a box for association at the next frame."""
        self.box = self.box_filter.predict(delta)
        self.age += 1

    def observe(
        self,
        box: NDArray[np.float64],
        confidence: float,
        class_id: int,
        metadata: dict[str, Any],
        mask: NDArray[Any] | None,
        keypoints: NDArray[Any] | None,
    ) -> None:
        """Correct the track from a direct detector observation."""
        self.box = self.box_filter.update(box)
        if self.config.confidence_stabilization:
            alpha = self.config.confidence_alpha
            self.confidence = alpha * confidence + (1.0 - alpha) * self.confidence
        else:
            self.confidence = confidence
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.missing = 0
        self.observations += 1
        self.uncertainty = max(self.config.initial_uncertainty, self.uncertainty * 0.5)
        switched, previous = self._observe_class(class_id, confidence)
        self.confirmed = self.observations >= self.config.min_confirmed_observations
        if not self.confirmed:
            self.source = Provenance.TENTATIVE
        elif self.config.box_smoothing == BoxSmoothing.NONE:
            self.source = Provenance.DIRECT
        else:
            self.source = Provenance.SMOOTHED
        self.metadata = dict(metadata)
        self.metadata.update(
            {
                "class_switch": switched,
                "previous_class_id": previous if switched else None,
                "observed": True,
            }
        )
        self.mask = None if mask is None else np.array(mask, copy=True)
        self.keypoints = None if keypoints is None else np.array(keypoints, copy=True)

    def mark_missing(self) -> None:
        """Advance prediction-only confidence and uncertainty."""
        self.missing += 1
        self.confidence = min(
            self.confidence, self.confidence * self.config.confidence_decay
        )
        self.uncertainty = min(1.0, self.uncertainty + self.config.uncertainty_growth)
        self.source = (
            Provenance.RECOVERED if self.missing == 1 else Provenance.PREDICTED
        )
        self.metadata = {
            **self.metadata,
            "class_switch": False,
            "previous_class_id": None,
            "observed": False,
        }
        self.mask = None
        self.keypoints = None

    def _observe_class(self, observed: int, confidence: float) -> tuple[bool, int]:
        previous = self.class_id
        if not self.config.class_stabilization:
            self.class_id = observed
            return self.class_id != previous, previous
        for class_id in self.class_evidence:
            self.class_evidence[class_id] *= self.config.class_evidence_decay
        self.class_evidence[observed] = (
            self.class_evidence.get(observed, 0.0) + confidence
        )
        self.class_history.append((observed, confidence))
        if self.config.class_voting == ClassVoting.MAJORITY:
            scores: dict[int, float] = {}
            for item, _score in self.class_history:
                scores[item] = scores.get(item, 0.0) + 1.0
        else:
            scores = dict(self.class_evidence)
        candidate = min(
            scores,
            key=lambda item: (-scores[item], item),
        )
        current_score = scores.get(self.class_id, 0.0)
        if (
            candidate != self.class_id
            and scores[candidate] >= current_score * self.config.class_switch_threshold
        ):
            self.class_id = candidate
        return self.class_id != previous, previous


@dataclass(slots=True)
class _StreamState:
    tracks: list[_Track] = field(default_factory=list)
    next_track_id: int = 1
    last_timestamp: float | None = None


class TemporalRepairer:
    """Repair frame-level detections while isolating independent streams."""

    def __init__(
        self,
        config: TemporalFixConfig | None = None,
        **overrides: Any,
    ) -> None:
        """Create a repairer from a config and optional explicit overrides."""
        if config is not None and overrides:
            raise ValueError("pass either config or keyword options, not both")
        self.config = (
            TemporalFixConfig.from_dict(overrides)
            if overrides
            else config or TemporalFixConfig()
        )
        self._streams: dict[str, _StreamState] = {}

    def reset(self, stream_id: str | None = None) -> None:
        """Reset one stream or all streams."""
        if stream_id is None:
            self._streams.clear()
        else:
            self._streams.pop(stream_id, None)

    def update(
        self,
        detections: Detections,
        *,
        frame: Any | None = None,
        timestamp: float | None = None,
        stream_id: str = "default",
    ) -> Detections:
        """Update one stream and return confirmed or requested tentative rows.

        ``frame`` is reserved for future motion compensation and is not read in
        version 0.1. Out-of-order finite timestamps are rejected. Equal
        timestamps use a unit prediction step to remain deterministic.
        """
        del frame
        state = self._streams.setdefault(stream_id, _StreamState())
        delta = self._time_delta(state, timestamp)
        for track in state.tracks:
            track.prepare(delta)

        active_detections = np.flatnonzero(
            detections.confidence >= self.config.min_confidence
        )
        candidates = detections[active_detections.astype(np.int64)]
        track_boxes = (
            np.stack([track.box for track in state.tracks])
            if state.tracks
            else np.empty((0, 4), dtype=np.float64)
        )
        track_classes = np.asarray(
            [track.class_id for track in state.tracks], dtype=np.int64
        )
        association = associate(
            track_boxes,
            candidates.xyxy,
            track_classes=track_classes,
            detection_classes=candidates.class_id,
            minimum_iou=self.config.min_iou,
            class_gating=self.config.class_gating,
        )

        for track_index, candidate_index in association.matches:
            track = state.tracks[track_index]
            mask = (
                None
                if candidates.masks is None
                else np.asarray(candidates.masks[candidate_index])
            )
            keypoints = (
                None
                if candidates.keypoints is None
                else np.asarray(candidates.keypoints[candidate_index])
            )
            track.observe(
                candidates.xyxy[candidate_index],
                float(candidates.confidence[candidate_index]),
                int(candidates.class_id[candidate_index]),
                dict(candidates.metadata[candidate_index]),
                mask,
                keypoints,
            )
        for track_index in association.unmatched_tracks:
            state.tracks[track_index].mark_missing()
        for candidate_index in association.unmatched_detections:
            self._create_track(state, candidates, candidate_index)

        state.tracks = [
            track
            for track in state.tracks
            if track.missing <= self.config.max_missing_frames
            and not (not track.confirmed and track.age > self.config.max_tentative_age)
        ]
        output = [
            track
            for track in sorted(state.tracks, key=lambda item: item.track_id)
            if track.confidence >= self.config.min_confidence
            and (
                track.confirmed
                or not self.config.suppress_short_tracks
                or self.config.output_tentative
            )
        ]
        return self._to_detections(output)

    @staticmethod
    def _time_delta(state: _StreamState, timestamp: float | None) -> float:
        if timestamp is None:
            return 1.0
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if state.last_timestamp is not None and timestamp < state.last_timestamp:
            raise ValueError("timestamp must not move backwards within a stream")
        delta = (
            1.0
            if state.last_timestamp is None or timestamp == state.last_timestamp
            else timestamp - state.last_timestamp
        )
        state.last_timestamp = timestamp
        return float(delta)

    def _create_track(
        self, state: _StreamState, detections: Detections, index: int
    ) -> None:
        box = detections.xyxy[index].copy()
        mask = None if detections.masks is None else np.asarray(detections.masks[index])
        keypoints = (
            None
            if detections.keypoints is None
            else np.asarray(detections.keypoints[index])
        )
        track = _Track(
            track_id=state.next_track_id,
            box_filter=_make_filter(box, self.config),
            box=box,
            confidence=float(detections.confidence[index]),
            class_id=int(detections.class_id[index]),
            config=self.config,
            uncertainty=self.config.initial_uncertainty,
            metadata={
                **dict(detections.metadata[index]),
                "class_switch": False,
                "previous_class_id": None,
                "observed": True,
            },
            mask=None if mask is None else np.array(mask, copy=True),
            keypoints=None if keypoints is None else np.array(keypoints, copy=True),
        )
        state.next_track_id += 1
        state.tracks.append(track)

    @staticmethod
    def _to_detections(tracks: list[_Track]) -> Detections:
        if not tracks:
            return Detections.empty()
        masks = (
            np.stack([track.mask for track in tracks if track.mask is not None])
            if all(track.mask is not None for track in tracks)
            else None
        )
        keypoints = (
            np.stack(
                [track.keypoints for track in tracks if track.keypoints is not None]
            )
            if all(track.keypoints is not None for track in tracks)
            else None
        )
        return Detections(
            xyxy=np.stack([track.box for track in tracks]),
            confidence=np.asarray(
                [track.confidence for track in tracks], dtype=np.float64
            ),
            class_id=np.asarray([track.class_id for track in tracks], dtype=np.int64),
            track_id=np.asarray([track.track_id for track in tracks], dtype=np.int64),
            masks=masks,
            keypoints=keypoints,
            metadata=[track.metadata for track in tracks],
            source=[track.source for track in tracks],
            uncertainty=np.asarray(
                [track.uncertainty for track in tracks], dtype=np.float64
            ),
            age=np.asarray([track.age for track in tracks], dtype=np.int64),
            time_since_update=np.asarray(
                [track.missing for track in tracks], dtype=np.int64
            ),
            is_confirmed=np.asarray(
                [track.confirmed for track in tracks], dtype=np.bool_
            ),
        )
