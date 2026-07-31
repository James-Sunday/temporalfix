# Command-line interface

```bash
temporalfix inspect-config --preset balanced
temporalfix validate-config config.yaml
temporalfix process-video input.mp4 --detections predictions.json
temporalfix benchmark benchmark.yaml
temporalfix version
```

`process-video` validates that the video path exists but consumes an ordered,
detector-independent JSON prediction file; it does not open the video or run a
detector. The JSON root is a list or `{"frames": [...]}`. Each frame contains
`detections` in `Detections.to_dict()` form and may include `timestamp` and
`stream_id`.

Use `temporalfix COMMAND --help` for the exact options. Validation and runtime
errors return a non-zero process status and a concise message on stderr.
