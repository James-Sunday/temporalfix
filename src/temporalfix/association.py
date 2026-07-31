"""Deterministic global detection association."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from temporalfix.geometry import pairwise_iou


@dataclass(frozen=True, slots=True)
class Association:
    """Matched and unmatched row indices."""

    matches: tuple[tuple[int, int], ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]


def _linear_sum_assignment(
    cost: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Solve rectangular minimum-cost assignment using Hungarian potentials."""
    if cost.ndim != 2:
        raise ValueError("cost must be a matrix")
    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    transposed = rows > columns
    matrix = cost.T if transposed else cost
    row_count, column_count = matrix.shape
    u = np.zeros(row_count + 1, dtype=np.float64)
    v = np.zeros(column_count + 1, dtype=np.float64)
    matched_row = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        min_value = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=np.bool_)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                current = (
                    matrix[current_row - 1, candidate - 1]
                    - u[current_row]
                    - v[candidate]
                )
                if current < min_value[candidate]:
                    min_value[candidate] = current
                    predecessor[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    row_indices = []
    column_indices = []
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            row_indices.append(int(matched_row[column] - 1))
            column_indices.append(column - 1)
    row_array = np.asarray(row_indices, dtype=np.int64)
    column_array = np.asarray(column_indices, dtype=np.int64)
    if transposed:
        return column_array, row_array
    return row_array, column_array


def associate(
    track_boxes: NDArray[np.float64],
    detection_boxes: NDArray[np.float64],
    *,
    track_classes: NDArray[np.int64],
    detection_classes: NDArray[np.int64],
    minimum_iou: float,
    class_gating: bool,
) -> Association:
    """Associate rows by globally minimizing gated ``1 - IoU`` cost."""
    track_count = len(track_boxes)
    detection_count = len(detection_boxes)
    if track_count == 0 or detection_count == 0:
        return Association(
            matches=(),
            unmatched_tracks=tuple(range(track_count)),
            unmatched_detections=tuple(range(detection_count)),
        )
    iou = pairwise_iou(track_boxes, detection_boxes)
    valid = iou >= minimum_iou
    if class_gating:
        known = (track_classes[:, None] >= 0) & (detection_classes[None, :] >= 0)
        valid &= ~known | (track_classes[:, None] == detection_classes[None, :])
    impossible = float(max(track_count, detection_count) + 2)
    cost = np.where(valid, 1.0 - iou, impossible)
    # A tiny stable term makes otherwise equal solutions deterministic.
    stable = (
        np.arange(track_count, dtype=np.float64)[:, None] * detection_count
        + np.arange(detection_count, dtype=np.float64)[None, :]
    )
    cost = cost + stable * np.finfo(np.float64).eps
    rows, columns = _linear_sum_assignment(cost)
    matches = tuple(
        sorted(
            (
                (int(row), int(column))
                for row, column in zip(rows, columns, strict=True)
                if valid[row, column]
            ),
            key=lambda item: item[0],
        )
    )
    matched_tracks = {item[0] for item in matches}
    matched_detections = {item[1] for item in matches}
    return Association(
        matches=matches,
        unmatched_tracks=tuple(
            index for index in range(track_count) if index not in matched_tracks
        ),
        unmatched_detections=tuple(
            index for index in range(detection_count) if index not in matched_detections
        ),
    )
