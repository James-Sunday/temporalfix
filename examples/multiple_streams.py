"""Keep two camera streams independent in one repairer."""

from __future__ import annotations

from temporalfix import Detections, TemporalFixConfig, TemporalRepairer


def observed(x: float) -> Detections:
    """Build one direct observation."""
    return Detections([[x, 0.0, x + 10.0, 10.0]], [0.9], [1])


def main() -> None:
    """Update and reset stream state independently."""
    repairer = TemporalRepairer(TemporalFixConfig.preset("low_latency"))
    camera_1 = repairer.update(observed(0.0), stream_id="camera_1")
    camera_2 = repairer.update(observed(100.0), stream_id="camera_2")
    assert camera_1.xyxy[0, 0] < 10
    assert camera_2.xyxy[0, 0] > 90

    repairer.reset(stream_id="camera_1")
    camera_1_restarted = repairer.update(observed(1.0), stream_id="camera_1")
    camera_2_continued = repairer.update(observed(101.0), stream_id="camera_2")
    assert camera_1_restarted.age.tolist() == [1]
    assert camera_2_continued.age.tolist() == [2]
    print(camera_1_restarted.to_dict())
    print(camera_2_continued.to_dict())


if __name__ == "__main__":
    main()
