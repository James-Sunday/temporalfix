# TemporalFix external benchmark protocol

This protocol separates detector quality from temporal-repair behavior. It
does not authorize redistribution of datasets, detector weights or derived
frames.

## Dataset and licence gate

The initial candidate is the MOT17 training split from the official
MOTChallenge site. MOTChallenge publishes its datasets under
CC BY-NC-SA 3.0, which restricts use to non-commercial purposes and requires
attribution and share-alike for adaptations. Users download it themselves,
review its current terms and record a content-addressed local manifest:

```bash
uv run python scripts/prepare_temporalfix_dataset.py path/to/MOT17 \
  --accept-license CC-BY-NC-SA-3.0 \
  --output benchmark_artifacts/datasets/mot17-manifest.json
```

The preparation command hashes files but does not copy, upload or redistribute
them. Cite MOTChallenge and every original sequence source requested by its
official data page.

## Compared systems

Run the same ordered frames and raw detections through:

1. raw detector output;
2. detector plus the selected tracker baseline;
3. detector plus TemporalFix;
4. detector plus tracker plus TemporalFix, only where their responsibilities
   and ID semantics make that composition valid.

Freeze detector outputs before comparison so every downstream system sees the
same observations. Record detector/runtime versions, all configuration,
sequence names, input checksums, random seeds and wall-clock environment.

## Metrics

- Box jitter: robust frame-to-frame displacement after subtracting annotated
  or fitted object motion.
- Class-switch rate: changed predicted class within a matched ground-truth
  trajectory divided by eligible transitions.
- Gap-recovery precision/recall: recovered rows matched to hidden
  ground-truth observations using a declared IoU threshold.
- False-positive persistence: duration distribution of unmatched output
  tracks.
- Detector accuracy: task metrics before and after repair, never inferred from
  temporal stability alone.
- Added latency: median, P95 and raw per-frame samples for the downstream
  processing stage.

Report sequence-level values and aggregate them with both macro and
frame-weighted summaries. Keep direct, recovered and predicted outputs
separate in error analysis.

## Guardrails

- Tune on training subdivisions, not the held-out test server.
- Do not call heuristic uncertainty calibrated.
- Do not claim improvement unless the stored artifact supports the exact
  metric and configuration.
- Publish no dataset, predictions or benchmark artifacts without checking
  their licences and obtaining repository-owner approval.

## Experiment configuration and raw schema

Every run records the resolved TemporalFix configuration, comparison system,
sequence/manifest hashes, seed, environment, package versions, per-frame raw
latency, per-sequence metrics, and aggregate method. Missing measurements are
null with a reason; they are never filled with a claimed improvement.

`analyze_artifact.py` creates tables and `plot_scaling.py` creates SVG only
from stored machine-readable JSON. The synthetic scaling entry point is the
initial reviewable experiment configuration; external configurations remain
paired with their user-managed dataset manifest.

## Hypothesis, threats, and paper outline

The working hypothesis is that short-term repair can reduce jitter, switches,
and short gaps at acceptable added latency without materially reducing task
accuracy. Dense crossings, camera motion, detector/domain bias, heuristic
uncertainty, tuning/test leakage, tracker choice, and non-commercial dataset
terms threaten generalization.

Working paper title: *TemporalFix: Detector-Agnostic Temporal Repair and
Uncertainty Estimation for Video Object Detection*. Planned sections are
motivation, related work, algorithm, reproducible protocol, experiments,
failure analysis, threats, and conclusions. Tables and figures must be derived
from artifacts; this outline contains no invented result.
