# Public API

The compatibility-protected 0.1 surface is exported from `temporalfix`:

```python
from temporalfix import (
    BoxSmoothing,
    ClassVoting,
    Detections,
    Provenance,
    TemporalFixConfig,
    TemporalRepairer,
    load_config,
)
```

`Detections` fields are `xyxy`, `confidence`, `class_id`, `track_id`, `masks`,
`keypoints`, `metadata`, `source`, `uncertainty`, `age`,
`time_since_update`, and `is_confirmed`. It supports `empty`, indexing,
`copy`, `to_dict`, and `from_dict`.

`TemporalRepairer.update(detections, timestamp=None, stream_id="default")`
updates one isolated stream. Timestamps must be finite and non-decreasing per
stream. `reset(stream_id=...)` resets one stream and `reset()` resets all.

`Provenance` values are `DIRECT`, `SMOOTHED`, `RECOVERED`, `PREDICTED`, and
`TENTATIVE`. Box strategies are `none`, `ema`, and `kalman`; class voting is
`majority` or `confidence_weighted`.
