# Five-minute quick start

```python
import numpy as np

from temporalfix import Detections, TemporalFixConfig, TemporalRepairer

repairer = TemporalRepairer(TemporalFixConfig.preset("balanced"))

for frame_index in range(5):
    raw = Detections(
        xyxy=np.asarray([[frame_index, 0, frame_index + 10, 10]]),
        confidence=np.asarray([0.9]),
        class_id=np.asarray([1]),
    )
    fixed = repairer.update(
        raw,
        timestamp=float(frame_index),
        stream_id="camera-1",
    )
    print(fixed.xyxy, fixed.track_id, fixed.source, fixed.uncertainty)
```

Inputs are copied, validated, and exposed as read-only arrays. XYXY uses
continuous half-open coordinates: width is `x2 - x1` and height is `y2 - y1`.
Empty detections are valid: `Detections.empty()`.

Run the complete examples from a source checkout:

```bash
uv run python examples/numpy_only.py
uv run python examples/multiple_streams.py
uv run python examples/missed_detection_recovery.py
uv run python examples/preset_comparison.py
```

Optional adapters are documented in [Optional integrations](integrations.md).
