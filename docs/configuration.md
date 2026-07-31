# Configuration

`TemporalFixConfig` is a frozen validated dataclass. Construct it directly,
load strict YAML with `load_config`, or resolve an inspectable preset.

```python
from temporalfix import BoxSmoothing, TemporalFixConfig

config = TemporalFixConfig(
    max_missing_frames=3,
    min_iou=0.35,
    box_smoothing=BoxSmoothing.EMA,
    output_tentative=True,
)
print(config.to_dict())
```

Available presets are `balanced`, `low_latency`, `high_stability`, and
`strict_false_positive_control`. No values are hidden:

```bash
temporalfix inspect-config --preset high_stability --format yaml
```

## Fields

| Area | Fields |
| --- | --- |
| Association | `min_iou`, `class_gating` |
| Gap lifecycle | `max_missing_frames` |
| Box smoothing | `box_smoothing`, `ema_alpha`, `kalman_process_noise`, `kalman_measurement_noise` |
| Confidence | `confidence_stabilization`, `confidence_alpha`, `confidence_decay`, `min_confidence` |
| Class evidence | `class_stabilization`, `class_voting`, `class_history_size`, `class_evidence_decay`, `class_switch_threshold` |
| Confirmation | `suppress_short_tracks`, `min_confirmed_observations`, `max_tentative_age`, `output_tentative` |
| Uncertainty | `initial_uncertainty`, `uncertainty_growth` |

YAML is limited to 1 MiB, parsed with `yaml.safe_load`, and must contain a
mapping. Unknown fields and invalid ranges raise clear errors.
