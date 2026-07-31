# TemporalFix

[![CI](https://github.com/James-Sunday/temporalfix/actions/workflows/ci.yml/badge.svg)](https://github.com/James-Sunday/temporalfix/actions/workflows/ci.yml)
[![Documentation](https://github.com/James-Sunday/temporalfix/actions/workflows/docs.yml/badge.svg)](https://James-Sunday.github.io/temporalfix/)

TemporalFix stabilizes and repairs frame-level object detections without
locking an application to a detector or full tracking framework.

> Status: `0.1.0rc1` is published and validated on
> [TestPyPI](https://test.pypi.org/project/temporalfix/0.1.0rc1/). Production
> PyPI remains gated on a stable `0.1.0` version and explicit owner approval.

## Installation

```bash
pip install temporalfix
```

Optional integrations are isolated:

```bash
pip install "temporalfix[opencv]"
pip install "temporalfix[supervision]"
pip install "temporalfix[ultralytics]"
```

Ultralytics uses AGPL-3.0 or a commercial licence; review its terms before
installing that extra.

## Five-minute quick start

```python
import numpy as np

from temporalfix import Detections, TemporalFixConfig, TemporalRepairer

repairer = TemporalRepairer(TemporalFixConfig.preset("balanced"))

for frame_index in range(5):
    detections = Detections(
        xyxy=np.asarray([[frame_index, 0, frame_index + 10, 10]]),
        confidence=np.asarray([0.9]),
        class_id=np.asarray([1]),
    )
    fixed = repairer.update(
        detections,
        timestamp=float(frame_index),
        stream_id="camera-1",
    )
    print(fixed.xyxy, fixed.track_id, fixed.source, fixed.uncertainty)
```

The complete NumPy example is `examples/numpy_only.py` and is executed during
verification.

```text
detector -> Detections -> global association -> stabilization
                                             -> provenance-labelled output
```

## Core features

- validated, owned, read-only NumPy arrays and JSON-compatible serialization;
- deterministic global IoU assignment with optional class gating;
- no smoothing, EMA smoothing and constant-velocity Kalman smoothing;
- observation-aware confidence smoothing and prediction-only decay;
- majority or confidence-weighted class evidence and switch diagnostics;
- bounded gap recovery with `RECOVERED`/`PREDICTED` provenance;
- configurable false-positive confirmation and optional tentative output;
- independent multi-stream state and scoped/global reset;
- strict safe YAML and inspectable presets.

## CLI

```bash
temporalfix inspect-config --preset balanced
temporalfix validate-config config.yaml
temporalfix process-video input.mp4 --detections predictions.json
temporalfix benchmark benchmark.yaml
temporalfix version
```

`process-video` reads detector-independent JSON and does not run or require a
detector. Use `--help` for schemas/options. Real failures return non-zero.

## Uncertainty and provenance

Uncertainty is a bounded `[0, 1]` heuristic, not a calibrated probability.
Direct observations reduce it toward `initial_uncertainty`; each missed frame
adds `uncertainty_growth` up to 1.0. Prediction-only frames also decay
confidence and never masquerade as detector observations.

Provenance is the `Provenance` enum: `DIRECT`, `SMOOTHED`, `RECOVERED`,
`PREDICTED`, and `TENTATIVE`.

## Benchmark methodology and results

The synthetic suite records warm-up count, every measured latency sample,
median/P95, seed, environment, resolved configuration and a separate
`tracemalloc` peak at 10/50/100/500 detections. Input generation is outside the
timed region. No portable performance or accuracy number is claimed here:
local artifacts are machine-specific, git-ignored and must be regenerated.

| Verified check | Observed scope | Claim boundary |
| --- | --- | --- |
| NumPy core, scenarios, CLI, and adapter contracts | Local Windows / Python 3.13 | Functional, not portable performance |
| Optional adapters | Ubuntu / Python 3.13 CI with installed extras | Supervision and Ultralytics contract tests passed |
| TestPyPI wheel | Clean GitHub runner and isolated local execution | Import, CLI, and minimal API smoke checks passed |

## Optional adapters

```python
from temporalfix.adapters import (
    from_supervision,
    from_ultralytics,
    to_supervision,
    to_ultralytics,
)
```

Imports remain dependency-light until a conversion function is called.
Supervision's public `data` mapping carries namespaced provenance, uncertainty
and lifecycle fields for lossless round trips. Ultralytics `Results` has no
documented equivalent; conversion back therefore raises on non-default
TemporalFix-only fields unless `allow_lossy=True` is explicit.

## Scope and limitations

TemporalFix is not an appearance-based tracker or long-term re-identification
system. Dense crossings, abrupt motion, camera motion and long occlusions can
change identity. Masks/keypoints are preserved for observed rows but are not
predicted across gaps.

## Contributing and citation

Development requires Ruff, strict Mypy, Pytest, strict documentation, security
scans, and clean wheel tests. See [the contributor guide](CONTRIBUTING.md),
[security policy](SECURITY.md), and [documentation](docs/index.md). Cite
[`CITATION.cff`](CITATION.cff); release history is in
[`CHANGELOG.md`](CHANGELOG.md). Original code is Apache-2.0 licensed.
