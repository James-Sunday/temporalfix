# TemporalFix changelog

## Unreleased

## 0.1.0 - 2026-07-31

### Changed

- Promoted the TestPyPI-validated release candidate to the stable version
  without runtime API changes.
- Hardened release automation with version-aware validation, current
  GitHub-maintained actions, a production clean-install check, and GitHub
  release creation only after the PyPI package validates.

## 0.1.0rc1 - 2026-07-31

### Added

- PEP 517 package foundation and version CLI.
- Validated `Detections`, provenance and lossless NumPy conversion.
- Deterministic IoU/Hungarian association and track lifecycle.
- None, EMA and Kalman box stabilization.
- Confidence/class stabilization, gap recovery and heuristic uncertainty.
- Multi-stream reset, strict YAML, inspectable presets and functional CLI.
- Unit, property, synthetic scenario and golden regression coverage.
- Lazy NumPy, Supervision and Ultralytics adapters with explicit loss handling.
- Executable multi-stream, class, gap, preset and optional-integration examples.
- Synthetic scaling/allocation benchmark, artifact analysis and SVG plotting.
- Licence-gated MOT17 manifest preparation and external benchmark protocol.
