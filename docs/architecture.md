# Architecture

TemporalFix separates the public data contract, configuration, association,
filters, lifecycle state, and optional adapters.

```text
Detections + config
       |
       v
pairwise IoU -> gated global assignment
       |                 |
       v                 v
 matched update     unmatched prediction
       |                 |
       +------ lifecycle/provenance ------+
                         |
                         v
                  immutable Detections
```

Association minimizes gated `1 - IoU` cost globally. Matched tracks receive a
filter correction and direct class/confidence evidence. Unmatched tracks
predict, decay confidence, increase uncertainty, and expire after
`max_missing_frames`. Expired identities never resurrect.

Each stream owns its tracks, IDs, and timestamp history. Optional framework
types are confined to adapters and never enter the core state model.
