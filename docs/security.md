# Security

Treat YAML, JSON, file paths, and optional native frameworks as untrusted input.
TemporalFix uses safe YAML parsing, file-size limits, explicit validation, and
no pickle-based configuration. User Python callbacks and framework objects run
with the caller's privileges.

Do not report vulnerabilities in a public issue. Follow the private process in
the repository-root `SECURITY.md`.
