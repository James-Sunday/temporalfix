# TemporalFix development guide

## Purpose and boundaries

TemporalFix is a detector-agnostic Python package for stabilising and repairing
frame-level video detections. The base installation stays NumPy-first and must
not require detector frameworks, OpenCV, ONNX, Ultralytics, or Supervision.

## Supported environment

- Python 3.11, 3.12, 3.13, and 3.14.
- Linux, Windows, and macOS.
- `uv` for locked development and release environments.

## Setup and commands

Run commands from the repository root.

```bash
uv sync --locked --all-groups
uv pip install -e .

uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov --cov-report=term-missing
uv run mkdocs build --strict
uv run bandit -r src
uv run detect-secrets scan --baseline .secrets.baseline
uv run pip-audit

uv run python benchmarks/synthetic_scaling.py
uv run python scripts/mutation_smoke.py
uv build
uv run twine check dist/*
uv run python scripts/verify_release_artifacts.py
uv run python scripts/smoke_install.py
```

## Engineering rules

- Keep public and core APIs typed, documented, deterministic, and testable.
- Validate untrusted YAML with safe loaders and reject unknown configuration.
- Validate paths, file sizes, timestamps, and subprocess arguments.
- Preserve lazy optional integrations and a dependency-light base import.
- Keep public interchange data NumPy-based.
- Add focused tests for behavior changes and run the relevant subset.
- Never suppress failures or warnings merely to make a gate green.
- Never claim a test, benchmark, hardware integration, or compatibility result
  unless it was executed and observed.
- Do not commit credentials, private paths, personal data, datasets without
  redistribution permission, or model weights.

## Completion and release restrictions

Before marking work complete, run the relevant format, lint, type, test,
coverage, docs, build, metadata, clean-install, example, audit, and secret
checks. Record unavailable optional-framework tests as skips with their reasons.

Never create a public repository, change visibility, publish to TestPyPI or
PyPI, create a public release, or upload benchmark artifacts without explicit
owner approval. Prefer Trusted Publishing with short-lived OIDC credentials;
never store release tokens in this repository.
