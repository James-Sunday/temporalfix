# Known limitations

- TemporalFix is not appearance-based tracking or long-term re-identification.
- Dense crossings, camera motion, abrupt movement, and long occlusion can
  change identity.
- Uncertainty is a deterministic bounded heuristic, not a calibrated
  probability.
- Masks and keypoints are carried for observations but not predicted in gaps.
- Ultralytics cannot losslessly represent TemporalFix-only lifecycle fields.
- `tracemalloc` measures Python allocation, not total process or device memory.
- Optional integrations are tested only when their dependencies are installed.
- Local compatibility covers the available Windows/Python interpreter; the CI
  matrix is the cross-platform gate for Python 3.11 through 3.14.
- The PyPI name check is point-in-time evidence, not a reservation or trademark
  clearance.
