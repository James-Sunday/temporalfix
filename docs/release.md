# Release process

1. Replace every ownership placeholder and confirm repository visibility.
2. Run `uv sync --locked --all-groups` and every command in `DEVELOPMENT.md`.
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

## Production release record

Stable version `0.1.0` contains no runtime API changes from the validated
release candidate. It was published through GitHub OIDC Trusted Publishing on
2026-07-31 by
[release run 30639578579](https://github.com/James-Sunday/temporalfix/actions/runs/30639578579)
from `main` commit `6e8c170a6c0ee67464914707bd2f46c154c21c4d`.

The run repeated every quality and release gate, uploaded both distributions,
installed `temporalfix==0.1.0` from production PyPI, ran import/version/CLI
probes, and created the public
[`v0.1.0` GitHub release](https://github.com/James-Sunday/temporalfix/releases/tag/v0.1.0)
only after package validation succeeded.

| Artifact | Production PyPI SHA-256 |
| --- | --- |
| `temporalfix-0.1.0-py3-none-any.whl` | `422388af90ecc08c9af98a185425364b06bf313f9728664a87460719026e4282` |
| `temporalfix-0.1.0.tar.gz` | `79d510b40de3075d3f9ed874793d9314391a6911c3729b998beeb96e3920d49e` |

An independent local Python 3.13 environment installed the exact published
wheel after verifying both downloaded artifact hashes. The version, CLI help,
and a minimal `Detections`/`TemporalRepairer` probe passed. PyPI records Trusted
Publishing attestations for both artifacts against the release workflow and
source commit.

The active production Trusted Publisher configuration is:

| Field | Value |
| --- | --- |
| PyPI project | `temporalfix` |
| GitHub owner | `James-Sunday` |
| Repository | `temporalfix` |
| Workflow | `release.yml` |
| Environment | `pypi` |

The PyPI `0.1.0` long description is immutable and therefore retains the
pre-publication status sentence embedded in its uploaded metadata. The current
repository README and documentation are authoritative; the corrected status
will be included in the next package version.
