# Release process

1. Replace every ownership placeholder and confirm repository visibility.
2. Run `uv sync --locked --all-groups` and every command in `AGENTS.md`.
3. Build once with `uv build`, validate metadata, inspect licences, and create
   checksums.
4. Publish the exact `0.1.0rc1` artifacts to TestPyPI through the protected
   `testpypi` environment using Trusted Publishing.
5. Install only from TestPyPI in a clean environment and run import, CLI, and
   minimal API smoke checks.
6. Record validation and obtain separate approval before changing to `0.1.0`
   and publishing production PyPI artifacts.

The release workflow requires a typed target confirmation and, for production,
an explicit TestPyPI-validation flag. Tags alone never publish.
