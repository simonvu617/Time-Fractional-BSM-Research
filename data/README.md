# Data Directory

No research dataset is committed to this repository.

```text
raw/        immutable vendor responses and canonical collection receipts
interim/    reproducible transformations that are not analysis-ready
processed/  frozen analysis panels consumed by experiments and tests
```

## Rules

- Raw vendor responses remain local unless redistribution rights are independently established.
- The MIT License covers this repository's original software; it does not relicense ThetaData data or other third-party material.
- Raw data are immutable. A refreshed pull receives a distinct versioned location rather than overwriting the evidence used by an earlier run.
- Interim and processed artifacts must be reproducible from raw inputs plus frozen code and configuration.
- Downstream models consume saved canonical outputs and must not call the vendor.
- Empty, missing, stale, or failed responses remain distinguishable in diagnostics.
- No dataset becomes acceptable merely because collection completed; timestamp semantics, schemas, coverage, missingness, and point-in-time behavior require an audit.
