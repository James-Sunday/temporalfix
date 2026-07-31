"""Bounding-box geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pairwise_iou(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute half-open pairwise IoU with zero for degenerate boxes."""
    if left.ndim != 2 or left.shape[1:] != (4,):
        raise ValueError("left boxes must have shape (N, 4)")
    if right.ndim != 2 or right.shape[1:] != (4,):
        raise ValueError("right boxes must have shape (M, 4)")
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection_size = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    left_area = np.maximum(0.0, left[:, 2] - left[:, 0]) * np.maximum(
        0.0, left[:, 3] - left[:, 1]
    )
    right_area = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(
        0.0, right[:, 3] - right[:, 1]
    )
    union = left_area[:, None] + right_area[None, :] - intersection
    result: NDArray[np.float64] = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    return result
