"""Base and optional detection adapters.

Importing this module does not import either optional framework.
"""

from temporalfix.adapters.numpy import from_numpy, to_numpy
from temporalfix.adapters.supervision import from_supervision, to_supervision
from temporalfix.adapters.ultralytics import from_ultralytics, to_ultralytics

__all__ = [
    "from_numpy",
    "from_supervision",
    "from_ultralytics",
    "to_numpy",
    "to_supervision",
    "to_ultralytics",
]
