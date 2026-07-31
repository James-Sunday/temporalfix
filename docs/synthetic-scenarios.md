# Synthetic scenario expectations

The deterministic core suite defines these expected behaviors:

1. Constant velocity: Kalman prediction advances in the observed direction.
2. Acceleration: geometry association retains the ID while configured gating
   permits the displacement.
3. Stationary jitter: EMA output position variance is below raw variance.
4. Two-object crossing: repeated runs produce identical assignments.
5. Short occlusion: the ID is retained, provenance is recovered/predicted,
   confidence falls and uncertainty rises.
6. Long occlusion: the old ID expires and re-entry creates a new ID.
7. One-frame false positive: confirmation suppresses it.
8. Temporary class switch: voting retains the old class until the configured
   evidence threshold and records a diagnostic at the actual switch.
9. Confidence collapse: observed smoothing dampens a single collapse, while
   prediction-only frames never increase confidence.
10. New object during a gap: the predicted old row and direct new row have
    distinct IDs.
11. Overlapping different classes: class gating prevents association when
    enabled.
12. Stream reset: state and ID allocation reset only for the selected stream,
    or for every stream with a global reset.
