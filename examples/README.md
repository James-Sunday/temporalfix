# TemporalFix examples

Run the dependency-light quick start:

```bash
python examples/numpy_only.py
```

The `configs/` directory exposes every resolved value for the balanced,
low-latency, high-stability and strict false-positive-control presets.

All examples are executable from a base development environment:

```bash
python examples/multiple_streams.py
python examples/class_stabilization.py
python examples/missed_detection_recovery.py
python examples/preset_comparison.py
python examples/supervision_adapter.py
python examples/ultralytics_adapter.py
```

The preset comparison covers both `low_latency` and `high_stability`.
Framework examples use only documented public fields. When their optional
extra is absent, they print the exact installation command and exit cleanly;
they do not download models or data.
