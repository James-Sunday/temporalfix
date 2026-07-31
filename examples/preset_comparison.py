"""Compare resolved low-latency and high-stability behavior."""

from __future__ import annotations

from temporalfix import Detections, TemporalFixConfig, TemporalRepairer


def main() -> None:
    """Run the same jittery observations through two explicit presets."""
    observations = [0.0, 1.3, 1.8, 3.4]
    for preset in ("low_latency", "high_stability"):
        config = TemporalFixConfig.preset(preset)
        repairer = TemporalRepairer(config)
        output_x1 = []
        for frame_index, x1 in enumerate(observations):
            raw = Detections([[x1, 0.0, x1 + 10.0, 10.0]], [0.9], [1])
            fixed = repairer.update(raw, timestamp=float(frame_index))
            output_x1.extend(fixed.xyxy[:, 0].tolist())
        print({"preset": preset, "resolved": config.to_dict(), "x1": output_x1})


if __name__ == "__main__":
    main()
