# Collector Contract

## Purpose

The collector creates the reproducible evidence base for the TFBSM study. Downstream analysis must consume saved canonical outputs and must not repeat vendor calls.

The unit of collection is one `symbol x trade_date`. The intended unit of later evaluation is:

```text
symbol x trade_date x evaluation_time x expiration x strike x right
```

## Module boundaries

| Module | Owns | Does not own |
| --- | --- | --- |
| `config` | configuration loading, path resolution, invariant checks | vendor requests or empirical choices not present in the config |
| `client` | read-only HTTP and CSV decoding | caching, normalization, selection, or retries hidden from the caller |
| `cache` | request identity, immutable raw payload, content hash, provenance metadata | canonical schemas or model-ready transformations |
| `timestamps` | exchange sessions, timezone localization, evaluation schedule | quote-quality thresholds or pricing time-to-maturity |
| `universe` | predeclared expiration and strike sampling | post-outcome sample filtering |
| `features` | mechanical quote fields and observed update/waiting-time measurements | BSM/TFBSM values, volatility fitting, or alpha selection |
| `validation` | required columns, parse failures, duplicates, count reconciliation | silently repairing vendor schema changes |
| `writer` | content-addressed canonical Parquet parts and manifests | deciding which rows support a hypothesis |
| `pipeline` | explicit orchestration of the modules above | hidden global state or research conclusions |

## Output layout

For the default configuration:

```text
data/raw/thetadata/
  raw/                         exact cached vendor CSV responses and metadata
  canonical/                   normalized, partitioned Parquet frames
  manifests/                   completed symbol-day receipts
```

Each raw cache entry records:

- endpoint and request parameters;
- request-identity digest;
- payload SHA-256 digest;
- fetch time and HTTP metadata;
- the exact CSV payload used by the collector.

Each canonical artifact records its dataset, partition, row count, path, and content digest. The symbol-day manifest links raw and canonical artifacts with validation results.

## Failure semantics

- Missing required vendor columns are errors, not empty data.
- A payload whose hash no longer matches its metadata is corrupt and is not reused.
- A different payload cannot overwrite an existing cache entry for the same request identity. A refreshed pull requires a new output directory or an explicitly designed versioned refresh workflow.
- Timestamps that cannot be parsed remain visible in validation counts.
- A symbol-day manifest is written only after the planned contract universe has been collected and reconciled.
- No request failure is converted into evidence of no trading or no liquidity.

## Research boundary

Collection establishes what the vendor returned and how it was mechanically normalized. It does not establish timestamp correctness, option style comparability, fillability, model superiority, a subdiffusive mechanism, or publishability. Those require separate audits and tests.

The point-in-time semantics of ThetaData's expiration and strike-list endpoints remain unverified for historical sessions because the configured list requests have no date parameter. That uncertainty blocks a multi-year run until a bounded vendor smoke test resolves it.
