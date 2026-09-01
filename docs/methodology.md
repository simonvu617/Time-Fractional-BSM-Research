# Methodology

## Research question

Does a correctly implemented subdiffusive time-fractional Black-Scholes model reduce out-of-sample option repricing error relative to classical BSM and credible alternative benchmarks, particularly in observations with direct evidence of slow market time?

The answer is unknown. Positive, negative, and underidentified results remain admissible.

## Unit of evaluation

```text
symbol x trade_date x evaluation_time x expiration x strike x right
```

The evaluation panel should preserve entry and exit quotes, underlying prices, maturity, rates, dividends, liquidity measures, waiting-time measures, sample flags, model inputs, train/test labels, and collection provenance.

## Pipeline

```text
collection
  -> data audit
  -> evaluation panel
  -> benchmark construction
  -> pricer validation
  -> rolling alpha selection
  -> out-of-sample comparison
  -> waiting-time diagnostics
  -> inference and robustness
  -> publication outputs
```

## Primary comparison

For observation `i`, define the primary loss as absolute exit repricing error. Define improvement as:

```text
d_i = loss_i(BSM) - loss_i(TFBSM)
```

A positive value favors TFBSM. The central conditional question is whether that improvement is larger in low-liquidity or high-waiting-time regimes. This definition does not establish that any improvement exists.

## Required discipline

- Lock the model and numerical solver before outcome testing.
- Use the same market inputs for BSM and TFBSM.
- Select `alpha` on training observations only.
- Apply selected parameters only to later observations.
- Preserve quote ages, spreads, quote-change flags, sample membership, exclusion reasons, and collection failures.
- Compare with entry-IV BSM, no-change, surface-IV BSM, liquidity-adjusted IV BSM, and historical-volatility BSM when supported by the audited data.
- Keep pricing improvement separate from the waiting-time evidence required for a structural mechanism claim.
- Address American exercise and dividend exposure explicitly.

## Current boundary

Only the collection foundation, model-domain specification, and basic repricing metrics exist. Pricers, calibration, backtests, inference, and empirical results are not implemented.
