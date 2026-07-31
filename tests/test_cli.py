from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporalfix.cli import main


def test_inspect_and_validate_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect-config", "--preset", "low_latency", "--format", "json"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["box_smoothing"] == "ema"

    config = tmp_path / "config.yaml"
    config.write_text("max_missing_frames: 2\n", encoding="utf-8")
    assert main(["validate-config", str(config)]) == 0
    assert "max_missing_frames: 2" in capsys.readouterr().out


def test_invalid_config_returns_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("unknown_option: true\n", encoding="utf-8")
    assert main(["validate-config", str(config)]) == 1
    assert "unknown configuration" in capsys.readouterr().err


def test_process_video_repairs_detector_independent_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"synthetic placeholder")
    predictions = tmp_path / "predictions.json"
    detection = {
        "xyxy": [[0, 0, 10, 10]],
        "confidence": [0.9],
        "class_id": [1],
    }
    predictions.write_text(
        json.dumps(
            {
                "frames": [
                    {"timestamp": 0.0, "detections": detection},
                    {"timestamp": 1.0, "detections": detection},
                    {
                        "timestamp": 2.0,
                        "detections": {"xyxy": [], "confidence": [], "class_id": []},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "repaired.json"
    assert (
        main(
            [
                "process-video",
                str(video),
                "--detections",
                str(predictions),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["frames"][0]["detections"]["xyxy"]) == 0
    assert result["frames"][1]["detections"]["track_id"] == [1]
    assert result["frames"][2]["detections"]["source"] == ["recovered"]
    assert "wrote 3 repaired frames" in capsys.readouterr().out


def test_benchmark_command_records_measured_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "frames: 3\nwarmup_frames: 1\ndetections_per_frame: 2\nseed: 7\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    assert main(["benchmark", str(config), "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["latency_ms"]["samples"]) == 3
    assert all(sample >= 0 for sample in result["latency_ms"]["samples"])
    assert result["configuration"]["seed"] == 7
    assert "wrote 3 measured" in capsys.readouterr().out
