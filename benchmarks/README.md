# TemporalFix benchmarks

Benchmarks write raw JSON artifacts under the root `benchmark_artifacts/`
directory. No performance threshold or result is asserted before a measured
baseline exists.

Run the required 10/50/100/500-row scaling suite:

```bash
uv run python benchmarks/synthetic_scaling.py \
  --counts 10 50 100 500 \
  --frames 20 \
  --warmup-frames 3 \
  --output benchmark_artifacts/temporalfix/synthetic-scaling.json
```

Input construction is outside the timed region. Each count receives its own
repairer and deterministic random generator; warm-up establishes active
tracks before measured frames. JSON records every latency sample, median,
P95, environment, resolved repair configuration, output row count and a
separate one-frame `tracemalloc` allocation peak.

`tracemalloc` is not process RSS, device memory or a cross-machine comparable
memory profiler. Artifacts are local and git-ignored by default. Derive tables
and an SVG without changing raw evidence:

```bash
uv run python research/analyze_artifact.py ARTIFACT.json
uv run python research/plot_scaling.py ARTIFACT.json \
  --output benchmark_artifacts/temporalfix/scaling.svg
```
