from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from temporalfix import BoxSmoothing, TemporalFixConfig, load_config


@pytest.mark.parametrize(
    "name",
    [
        "balanced",
        "low_latency",
        "high_stability",
        "strict_false_positive_control",
    ],
)
def test_presets_are_visible_and_valid(name: str) -> None:
    config = TemporalFixConfig.preset(name)
    assert set(config.to_dict()) == {
        field.name for field in __import__("dataclasses").fields(config)
    }


def test_unknown_and_invalid_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        TemporalFixConfig.from_dict({"typo": True})
    with pytest.raises(ValueError, match="min_iou"):
        TemporalFixConfig(min_iou=2.0)
    with pytest.raises(ValueError, match="unknown preset"):
        TemporalFixConfig.preset("fastest")


def test_yaml_load_is_strict_and_safe(tmp_path: Path) -> None:
    valid = tmp_path / "config.yaml"
    valid.write_text("box_smoothing: ema\nmax_missing_frames: 2\n", encoding="utf-8")
    config = load_config(valid)
    assert config.box_smoothing == BoxSmoothing.EMA
    assert config.max_missing_frames == 2

    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text("!!python/object/apply:os.system ['echo bad']", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(unsafe)


def test_configuration_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "large.yaml"
    path.write_text("x" * 1_048_577, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        load_config(path)
