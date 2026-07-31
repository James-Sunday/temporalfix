from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays

from temporalfix.association import _linear_sum_assignment, associate
from temporalfix.geometry import pairwise_iou


def test_pairwise_iou_handles_empty_and_degenerate_boxes() -> None:
    empty = np.empty((0, 4), dtype=np.float64)
    assert pairwise_iou(empty, empty).shape == (0, 0)
    result = pairwise_iou(
        np.asarray([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]),
        np.asarray([[1.0, 1.0, 3.0, 3.0]]),
    )
    assert result[0, 0] == pytest.approx(1 / 7)
    assert result[1, 0] == 0.0


def test_hungarian_matches_exhaustive_optimum() -> None:
    cost = np.asarray([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    rows, columns = _linear_sum_assignment(cost)
    observed = float(cost[rows, columns].sum())
    expected = min(
        sum(cost[row, column] for row, column in enumerate(permutation))
        for permutation in itertools.permutations(range(3))
    )
    assert observed == expected


def test_association_class_gating_is_configurable() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0]])
    gated = associate(
        boxes,
        boxes,
        track_classes=np.asarray([1]),
        detection_classes=np.asarray([2]),
        minimum_iou=0.5,
        class_gating=True,
    )
    ungated = associate(
        boxes,
        boxes,
        track_classes=np.asarray([1]),
        detection_classes=np.asarray([2]),
        minimum_iou=0.5,
        class_gating=False,
    )
    assert gated.matches == ()
    assert ungated.matches == ((0, 0),)


def test_equal_cost_assignment_is_deterministic() -> None:
    boxes = np.asarray([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    kwargs = {
        "track_classes": np.asarray([0, 0]),
        "detection_classes": np.asarray([0, 0]),
        "minimum_iou": 0.1,
        "class_gating": False,
    }
    first = associate(boxes, boxes, **kwargs)
    assert all(associate(boxes, boxes, **kwargs) == first for _ in range(10))
    assert first.matches == ((0, 0), (1, 1))


@given(
    boxes=arrays(
        np.float64,
        (5, 4),
        elements={"min_value": -1e3, "max_value": 1e3, "allow_nan": False},
    )
)
def test_iou_is_bounded_for_normalized_boxes(boxes: np.ndarray) -> None:
    normalized = np.column_stack(
        (
            np.minimum(boxes[:, 0], boxes[:, 2]),
            np.minimum(boxes[:, 1], boxes[:, 3]),
            np.maximum(boxes[:, 0], boxes[:, 2]),
            np.maximum(boxes[:, 1], boxes[:, 3]),
        )
    )
    result = pairwise_iou(normalized, normalized)
    assert np.all((result >= 0.0) & (result <= 1.0))
