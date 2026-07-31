# Release process

1. Replace every ownership placeholder and confirm repository visibility.
2. Run `uv sync --locked --all-groups` and every command in `AGENTS.md`.
3. Build once with `uv build`, validate metadata, inspect licences, and create
   checksums.
4. Publish the exact `0.1.0rc1` artifacts to TestPyPI through the protected
   `testpypi` environment using Trusted Publishing.
5. Install only from TestPyPI in a clean environment and run import, CLI, and
   minimal API smoke checks.
6. Record validation and obtain approval before preparing stable version
   `0.1.0`.
7. Re-run every gate against the stable version and review its draft release
   pull request.
8. Obtain separate approval before publishing to production PyPI or creating a
   public GitHub release.

The release workflow requires a typed target confirmation and, for production,
an explicit TestPyPI-validation flag. Tags alone never publish.

## TestPyPI validation record

TemporalFix `0.1.0rc1` was published through GitHub OIDC Trusted Publishing on
2026-07-31 and validated by
[release run 30636027408](https://github.com/James-Sunday/temporalfix/actions/runs/30636027408).
The run rebuilt from `main`, repeated all release gates, uploaded both formats,
installed the package from TestPyPI without dependencies, and ran import,
version, and CLI-help probes.

An independent local check installed the exact published wheel URL and ran a
minimal `Detections`/`TemporalRepairer` API probe. The published artifacts are:

| Artifact | TestPyPI SHA-256 |
| --- | --- |
| `temporalfix-0.1.0rc1-py3-none-any.whl` | `be75afed5ff1cd9231c9a2cd0164b01eea0ec8d732525244acec858ec1ff9e60` |
| `temporalfix-0.1.0rc1.tar.gz` | `4fc7045aaddb2a83dfcd49ae7b8b0b40f2ed8089eec464f68496dd093121949d` |

The Linux-published and locally built Windows wheels differ in ZIP container
metadata, but all 21 member paths and member SHA-256 values are identical. The
source-distribution SHA-256 is identical across the two builds.

## Production preparation

Stable version `0.1.0` contains no runtime API changes from the validated
release candidate. Before the production run, register this pending Trusted
Publisher at <https://pypi.org/manage/account/publishing/>:

| Field | Value |
| --- | --- |
| PyPI project name | `temporalfix` |
| GitHub owner | `James-Sunday` |
| Repository | `temporalfix` |
| Workflow | `release.yml` |
| Environment | `pypi` |

After the stable release pull request is merged and production publication is
explicitly approved, dispatch `release.yml` from `main` with target `pypi`,
confirmation `pypi`, and `testpypi_validated` enabled. The workflow rebuilds
and verifies the artifacts, publishes through OIDC, installs the published
version from PyPI, and creates `v0.1.0` only after that installation succeeds.

Production PyPI upload and the public GitHub release remain blocked until the
owner gives that separate approval.
