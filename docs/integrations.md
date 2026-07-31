# Optional integrations

The core import does not import OpenCV, ONNX, Supervision, or Ultralytics.

```bash
pip install "temporalfix[supervision]"
pip install "temporalfix[ultralytics]"
pip install "temporalfix[opencv]"
pip install "temporalfix[onnx]"
```

```python
from temporalfix.adapters import (
    from_supervision,
    from_ultralytics,
    to_supervision,
    to_ultralytics,
)
```

Supervision's public `data` mapping carries namespaced TemporalFix lifecycle
fields for lossless round trips. Ultralytics `Results` has no documented place
for provenance, uncertainty, age, confirmation, or arbitrary row metadata;
conversion back rejects non-default values unless `allow_lossy=True` is
explicit.

Both adapters handle empty detections and raise installation guidance if their
optional framework is absent. Ultralytics is AGPL-3.0 or commercially licensed;
review its terms before choosing that extra.
