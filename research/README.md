# TemporalFix research

Working question: can detector-agnostic temporal repair reduce box jitter,
class switching and short gaps without materially degrading detector accuracy
or adding unacceptable latency?

This directory contains:

- `benchmark_protocol.md`: comparison systems, metrics, licence gates and
  evidence requirements.
- `analyze_artifact.py`: Markdown tables derived from measured JSON.
- `plot_scaling.py`: dependency-free SVG plots derived from measured JSON.
- `CITATION.cff`: package-specific software citation metadata.

Dataset content, model weights and benchmark outputs are never bundled here.
The MOT17 preparation command works only on an existing user download after
exact CC BY-NC-SA 3.0 acknowledgement and stores relative content hashes
rather than private local paths. Experiment schemas, the hypothesis, threats
to validity, and the paper outline are recorded in the protocol and remain
evidence-based.
