# Accuracy and performance

Temporal stability and detector accuracy are different measurements. External
evaluation compares raw detections, a tracker baseline, TemporalFix, and a
valid tracker-plus-TemporalFix composition on identical frozen detector output.

The protocol measures box jitter, class-switch rate, gap-recovery precision and
recall, false-positive persistence, detector task metrics, and added latency.
No improvement is claimed unless a stored machine-readable artifact supports
the exact configuration, dataset split, metric, and environment.

Smoothing trades responsiveness for stability. `none` has no smoothing lag;
EMA is inexpensive and responds faster as `ema_alpha` increases; Kalman
smoothing predicts constant velocity and depends on its process and
measurement noise settings.
