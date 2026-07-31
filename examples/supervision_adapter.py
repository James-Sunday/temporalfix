"""Round-trip repaired detections through optional Supervision."""

from __future__ import annotations

import importlib

import numpy as np

from temporalfix import TemporalRepairer
from temporalfix.adapters import from_supervision, to_supervision


def main() -> None:
    """Run when Supervision is installed; otherwise explain the extra."""
    try:
        sv = importlib.import_module("supervision")
    except ModuleNotFoundError:
        print('Skipped: install with `pip install "temporalfix[supervision]"`.')
        return

    incoming = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([1]),
    )
    fixed = TemporalRepairer().update(from_supervision(incoming))
    outgoing = to_supervision(fixed)
    print(outgoing.xyxy, outgoing.tracker_id, outgoing.data)


if __name__ == "__main__":
    main()
