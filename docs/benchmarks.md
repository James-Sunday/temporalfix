# Benchmark methodology

The synthetic suite measures update-only latency at 10, 50, 100, and 500
detections per frame. Input generation is outside the timed region. It records
warm-up count, every measured sample, median, P95, seed, environment, resolved
configuration, and a separately traced allocation peak.

```bash
uv run python benchmarks/synthetic_scaling.py \
  --output benchmark_artifacts/temporalfix/synthetic-scaling.json
uv run python research/analyze_artifact.py \
  benchmark_artifacts/temporalfix/synthetic-scaling.json
uv run python research/plot_scaling.py \
  benchmark_artifacts/temporalfix/synthetic-scaling.json \
  --output benchmark_artifacts/temporalfix/scaling.svg
```

Local artifacts are machine-specific, git-ignored, and not portable claims.
The external protocol, licence gate, metrics, and raw-results schema are in the
repository's `research/benchmark_protocol.md`.
