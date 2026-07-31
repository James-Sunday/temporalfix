"""Show provenance, confidence and uncertainty during a short gap."""

from __future__ import annotations

from temporalfix import Detections, TemporalFixConfig, TemporalRepairer


def main() -> None:
    """Confirm a track, omit two frames, then observe it again."""
    repairer = TemporalRepairer(TemporalFixConfig.preset("balanced"))
    frames = [
        Detections([[0, 0, 10, 10]], [0.95], [1]),
        Detections([[1, 0, 11, 10]], [0.9], [1]),
        Detections.empty(),
        Detections.empty(),
        Detections([[4, 0, 14, 10]], [0.88], [1]),
    ]
    for frame_index, raw in enumerate(frames):
        fixed = repairer.update(raw, timestamp=float(frame_index))
        print(
            {
                "frame": frame_index,
                "source": [str(item) for item in fixed.source],
                "confidence": fixed.confidence.tolist(),
                "uncertainty": fixed.uncertainty.tolist(),
            }
        )


if __name__ == "__main__":
    main()
