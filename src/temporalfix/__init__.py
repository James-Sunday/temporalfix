"""Detector-agnostic temporal repair for video detections."""

from temporalfix._version import __version__
from temporalfix.config import (
    BoxSmoothing,
    ClassVoting,
    TemporalFixConfig,
    load_config,
)
from temporalfix.detections import Detections, Provenance
from temporalfix.repairer import TemporalRepairer

__all__ = [
    "BoxSmoothing",
    "ClassVoting",
    "Detections",
    "Provenance",
    "TemporalFixConfig",
    "TemporalRepairer",
    "__version__",
    "load_config",
]
