"""Convert optional Ultralytics ``Results`` through TemporalFix."""

from __future__ import annotations

import importlib

import numpy as np

from temporalfix import TemporalRepairer
from temporalfix.adapters import from_ultralytics, to_ultralytics


def main() -> None:
    """Run without model weights when Ultralytics is installed."""
    try:
        results_module = importlib.import_module("ultralytics.engine.results")
    except ModuleNotFoundError:
        print('Skipped: install with `pip install "temporalfix[ultralytics]"`.')
        return

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    incoming = results_module.Results(
        orig_img=image,
        path="synthetic-frame",
        names={0: "object"},
        boxes=np.array([[2, 3, 20, 22, 0.9, 0]], dtype=np.float32),
    )
    fixed = TemporalRepairer().update(from_ultralytics(incoming))
    # Ultralytics Results has no public provenance/uncertainty fields, so loss
    # must be explicitly acknowledged when converting repaired output back.
    outgoing = to_ultralytics(
        fixed,
        orig_img=image,
        names=incoming.names,
        allow_lossy=True,
    )
    print(outgoing.boxes.xyxy)


if __name__ == "__main__":
    main()
