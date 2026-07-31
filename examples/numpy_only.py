"""Minimal NumPy-only TemporalFix example."""

from __future__ import annotations

import numpy as np

from temporalfix import Detections, TemporalFixConfig, TemporalRepairer


def main() -> None:
    """Run three observed frames followed by a one-frame gap."""
    repairer = TemporalRepairer(TemporalFixConfig.preset("low_latency"))
    for frame_index in range(4):
        if frame_index == 3:
            detections = Detections.empty()
        else:
            x = float(frame_index)
            detections = Detections(
                xyxy=np.asarray([[x, 0.0, x + 10.0, 10.0]]),
                confidence=np.asarray([0.9]),
                class_id=np.asarray([1]),
            )
        fixed = repairer.update(detections, timestamp=float(frame_index))
        print(fixed.to_dict())


if __name__ == "__main__":
    main()
