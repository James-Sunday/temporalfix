"""Strict TemporalFix configuration and presets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml

MAX_CONFIG_BYTES = 1_048_576


class BoxSmoothing(StrEnum):
    """Supported box stabilization methods."""

    NONE = "none"
    EMA = "ema"
    KALMAN = "kalman"


class ClassVoting(StrEnum):
    """Supported class evidence aggregation methods."""

    MAJORITY = "majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass(frozen=True, slots=True)
class TemporalFixConfig:
    """Validated configuration for :class:`TemporalRepairer`."""

    max_missing_frames: int = 4
    min_iou: float = 0.3
    class_gating: bool = False
    box_smoothing: BoxSmoothing = BoxSmoothing.KALMAN
    ema_alpha: float = 0.65
    kalman_process_noise: float = 1.0
    kalman_measurement_noise: float = 4.0
    confidence_stabilization: bool = True
    confidence_alpha: float = 0.6
    confidence_decay: float = 0.85
    min_confidence: float = 0.05
    class_stabilization: bool = True
    class_voting: ClassVoting = ClassVoting.CONFIDENCE_WEIGHTED
    class_history_size: int = 8
    class_evidence_decay: float = 0.9
    class_switch_threshold: float = 1.2
    suppress_short_tracks: bool = True
    min_confirmed_observations: int = 2
    max_tentative_age: int = 3
    output_tentative: bool = False
    initial_uncertainty: float = 0.1
    uncertainty_growth: float = 0.15

    def __post_init__(self) -> None:
        """Validate all configuration values."""
        if self.max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        if not 0.0 <= self.min_iou <= 1.0:
            raise ValueError("min_iou must be within [0, 1]")
        for name in ("ema_alpha", "confidence_alpha", "confidence_decay"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if self.kalman_process_noise <= 0.0 or self.kalman_measurement_noise <= 0.0:
            raise ValueError("Kalman noise values must be positive")
        if self.class_history_size < 1:
            raise ValueError("class_history_size must be positive")
        if not 0.0 < self.class_evidence_decay <= 1.0:
            raise ValueError("class_evidence_decay must be within (0, 1]")
        if self.class_switch_threshold < 1.0:
            raise ValueError("class_switch_threshold must be at least 1")
        if self.min_confirmed_observations < 1:
            raise ValueError("min_confirmed_observations must be positive")
        if self.max_tentative_age < 1:
            raise ValueError("max_tentative_age must be positive")
        if not 0.0 <= self.initial_uncertainty <= 1.0:
            raise ValueError("initial_uncertainty must be within [0, 1]")
        if not 0.0 <= self.uncertainty_growth <= 1.0:
            raise ValueError("uncertainty_growth must be within [0, 1]")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Create a config while rejecting unknown fields."""
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown configuration options: {', '.join(unknown)}")
        normalized = dict(values)
        if "box_smoothing" in normalized:
            normalized["box_smoothing"] = BoxSmoothing(normalized["box_smoothing"])
        if "class_voting" in normalized:
            normalized["class_voting"] = ClassVoting(normalized["class_voting"])
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return visible JSON/YAML-compatible resolved values."""
        result = asdict(self)
        result["box_smoothing"] = str(self.box_smoothing)
        result["class_voting"] = str(self.class_voting)
        return result

    @classmethod
    def preset(cls, name: str) -> Self:
        """Resolve a named preset to ordinary explicit values."""
        presets = {
            "balanced": cls(),
            "low_latency": replace(
                cls(),
                box_smoothing=BoxSmoothing.EMA,
                ema_alpha=0.8,
                max_missing_frames=2,
                min_confirmed_observations=1,
            ),
            "high_stability": replace(
                cls(),
                ema_alpha=0.45,
                max_missing_frames=6,
                class_history_size=12,
                min_confirmed_observations=3,
            ),
            "strict_false_positive_control": replace(
                cls(),
                min_confirmed_observations=3,
                max_tentative_age=4,
                output_tentative=False,
                min_confidence=0.2,
            ),
        }
        try:
            return presets[name]
        except KeyError as error:
            raise ValueError(
                f"unknown preset {name!r}; choose from {', '.join(sorted(presets))}"
            ) from error


def load_config(path: str | Path) -> TemporalFixConfig:
    """Load a size-limited, safe YAML configuration."""
    config_path = Path(path)
    size = config_path.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ValueError(f"configuration exceeds {MAX_CONFIG_BYTES} bytes")
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, Mapping):
        raise ValueError("configuration root must be a mapping")
    return TemporalFixConfig.from_dict(parsed)
