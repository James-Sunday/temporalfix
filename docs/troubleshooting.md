# Troubleshooting and FAQ

## Why is the first observation absent?

Confirmed output is suppressed by default until
`min_confirmed_observations` is reached. Use the `low_latency` preset or enable
tentative output only when that trade-off is intended.

## Why did an identity change?

Association uses geometry and optional class gating, not appearance features.
Dense crossings, abrupt motion, and long gaps can create a new identity.

## Is uncertainty an error probability?

No. It describes observation/prediction lifecycle state and has not been
calibrated against empirical error coverage.

## Why does an adapter import fail?

Install its extra, for example `pip install "temporalfix[supervision]"`.
Importing `temporalfix` alone intentionally does not install or import optional
frameworks.

## Why was YAML rejected?

Configuration must be a mapping under 1 MiB and contain only documented fields
with valid ranges. This strictness prevents silent misspellings.
