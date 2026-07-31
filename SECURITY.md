# Security policy

## Supported versions

Until the first stable public release, the latest release-candidate line
receives security fixes. After 0.1.0, the latest minor release is supported.

## Reporting

Do not open public issues for suspected vulnerabilities. Use the repository's
[private vulnerability-reporting form](https://github.com/James-Sunday/temporalfix/security/advisories/new)
with the affected version, reproduction steps, impact, and suggested
mitigation.

## Security model

- YAML is parsed with safe loaders after file-size checks.
- Untrusted pickle and arbitrary Python deserialization are unsupported.
- Detector/framework objects are untrusted input; process them in a constrained
  environment when their provenance is unknown.
- User callbacks execute with the caller's process privileges and are trusted
  code.
- Paths and file sizes are validated before reads or writes.
- Optional native runtimes expand the attack surface and must be kept patched.

Dependency audit, static analysis and secret scanning are CI gates. Accepted
exceptions require an issue, rationale, owner and expiry date.
