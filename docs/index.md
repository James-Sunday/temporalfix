# TemporalFix

TemporalFix stabilizes and repairs frame-level detections without coupling an
application to a detector or full tracking framework. It accepts validated
NumPy data and returns boxes with track IDs, lifecycle state, provenance, and
bounded heuristic uncertainty.

```text
detector -> Detections -> global association -> temporal stabilization
                                             -> provenance-labelled output
```

Install the dependency-light core:

```bash
pip install temporalfix
```

Start with the [five-minute quick start](quickstart.md), then review the
[configuration](configuration.md), [public API](api.md), and
[limitations](known-limitations.md). The release candidate has no published
performance or accuracy claim; its benchmark tools record evidence for the
machine and data on which they run.

## What it does

- deterministic global IoU association with optional class gating;
- none, EMA, or constant-velocity Kalman box smoothing;
- confidence smoothing and decay through short gaps;
- class evidence voting and switch diagnostics;
- false-positive confirmation and bounded gap recovery;
- independent state for multiple streams;
- safe YAML configuration and inspectable presets.

## What it does not do

TemporalFix is not appearance-based tracking or long-term re-identification.
Its uncertainty is an interpretable lifecycle heuristic, not a calibrated
error probability. Detector execution remains outside the package.
