# Experiment 001: Collection Smoke Test

## Objective

Establish whether the modular collector can produce one complete, auditable SPY symbol-day before any multi-year collection or model comparison.

This experiment is an infrastructure and data-semantics test. It cannot establish that TFBSM improves repricing, that `alpha` differs from `1`, or that the inverse-stable mechanism is supported.

## Current status

`specified_not_run`

The synthetic integration test passes, but no live ThetaData request has been made. The expiration and strike-list endpoints' historical point-in-time semantics remain unresolved and are an explicit acceptance check.

## Dry run

From the repository root:

```bash
tfbsm-collect \
  --config configs/default.yaml \
  --symbol SPY \
  --date 2025-01-03 \
  --dry-run
```

The live form removes `--dry-run`. It should be run only when the local vendor process is intentionally available, and that process should be stopped afterward.
