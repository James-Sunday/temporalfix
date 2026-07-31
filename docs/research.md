# Research protocol

The working question is whether detector-agnostic temporal repair can reduce
box jitter, class switching, and short gaps without materially degrading task
accuracy or adding unacceptable latency.

The repository's `research/` directory contains the protocol, artifact
analysis, plot generation, citation metadata, hypotheses, threats to validity,
and paper outline. Dataset content, model weights, and result artifacts are not
bundled.

MOT17 preparation requires an existing user download and exact acknowledgement
of its CC BY-NC-SA 3.0 terms:

```bash
uv run python scripts/prepare_temporalfix_dataset.py path/to/MOT17 \
  --accept-license CC-BY-NC-SA-3.0 \
  --output benchmark_artifacts/datasets/mot17-manifest.json
```

The command hashes files into a relative manifest; it does not copy, download,
or upload the dataset.
