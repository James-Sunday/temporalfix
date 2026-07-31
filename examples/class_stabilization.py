"""Demonstrate confidence-weighted resistance to a one-frame class switch."""

from __future__ import annotations

from temporalfix import Detections, TemporalFixConfig, TemporalRepairer


def main() -> None:
    """Feed stable class evidence followed by one conflicting observation."""
    resolved = TemporalFixConfig.preset("high_stability").to_dict()
    resolved["output_tentative"] = True
    config = TemporalFixConfig.from_dict(resolved)
    repairer = TemporalRepairer(config)
    observed_classes = [3, 3, 8, 3]
    for frame_index, class_id in enumerate(observed_classes):
        raw = Detections(
            [[float(frame_index), 0.0, float(frame_index + 10), 10.0]],
            confidence=[0.9 if class_id == 3 else 0.55],
            class_id=[class_id],
        )
        fixed = repairer.update(raw, timestamp=float(frame_index))
        print(
            {
                "frame": frame_index,
                "observed_class": class_id,
                "stabilized_class": int(fixed.class_id[0]),
                "metadata": fixed.metadata[0],
            }
        )


if __name__ == "__main__":
    main()
