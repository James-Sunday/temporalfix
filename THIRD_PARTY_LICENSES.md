# Third-party licences

TemporalFix does not vendor third-party source, datasets, or model weights.
Runtime and development dependencies retain their own licences.

The base dependencies, NumPy and PyYAML, use permissive BSD/MIT licences.
Optional OpenCV, ONNX, and Supervision integrations are permissively licensed
at the reviewed versions. Ultralytics is offered under AGPL-3.0 or a commercial
licence; users must review the terms before installing that optional extra.

Dependency audit and licence inventory should be regenerated for each release.
MOT17 is never bundled; its external benchmark path requires the user to accept
CC BY-NC-SA 3.0.
