# Time-Fractional Black-Scholes Research

This repository is an empirical-research foundation for studying subdiffusive time-fractional Black-Scholes (TFBSM) across stock and ETF option markets and observation frequencies.

## Research question

**How do TFBSM's pricing accuracy and estimated fractional parameter alpha vary across stock and ETF option markets, observation frequencies, and market conditions, compared with classical Black-Scholes?**

The study extends the daily S&P 500 option analysis of [An et al. (2024)](https://doi.org/10.1007/s11075-023-01563-4) to ask whether its findings generalize across underlying assets and persist within the trading day. More observations provide the means to answer those questions; sample size alone is not the intended contribution.

Out-of-sample testing supports the pricing comparison by evaluating observations excluded from parameter estimation. Forecasting later prices and evaluating trading returns are possible extensions, rather than the primary research question. Repricing with an observed future underlying price must be distinguished from a forecast made before that price is known.

The answer is unknown. The project must support a positive, negative, or inconclusive result without changing the test after seeing outcomes.

## Candidate contribution

The strongest proposed focus is to test **how much fitted alpha and any pricing advantage reflect market inactivity, observation frequency, or stale and discretized quotes**. These are candidate experiments, not implemented analyses or established novelty claims:

1. **Compare observation frequencies on matched data.** Construct daily and intraday samples from the same underlying records. Keep contract/date coverage, elapsed calibration windows, model time units, and evaluation targets comparable; report coverage differences. Examine both alpha estimates and pricing errors. Changing frequency need not leave fitted alpha unchanged, and a change alone does not establish a sampling artifact.
2. **Test the interpretation against measured activity.** Examine whether alpha and pricing improvements are associated with trade gaps and observed price persistence after accounting for volatility, spreads, moneyness, and maturity. Report variation across underlying assets and market conditions. Sampled quote persistence, trade arrivals, and underlying inactivity are distinct measurements; their relationship to an option-implied pricing parameter requires testing.
3. **Use controls that can disprove the interpretation.** Apply the same observation and estimation procedure to simulated classical BSM data with rounding, stale observations, and missing records, and to simulated TFBSM data with known parameters. Determine when the procedure falsely reports fractional behavior and when it can recover it. Evaluate held-out prices against comparable non-fractional benchmarks so that extra fitting flexibility is not mistaken for a useful mechanism.

A useful result could establish robust cross-market patterns, identify the observation scales where the model adds information, or show that apparent benefits disappear after measurement controls. Exact frequencies, calibration windows, parameterization, benchmarks, primary metrics, and sample rules remain design choices to settle before evaluating the final comparison.

## Relationship to prior work

The starting literature already includes:

- [An et al. (2024)](https://doi.org/10.1007/s11075-023-01563-4): numerical solution, joint parameter estimation, and daily S&P 500 option-price fitting across market conditions.
- [Orzel and Weron (2010)](https://www.actaphys.uj.edu.pl/R/41/5/1151/pdf): calibration of subdiffusive BSM using constant-price durations and nonconstant observations.
- [Torricelli (2020)](https://doi.org/10.1016/j.physa.2019.123694): trade-duration risk and behavior across time scales in subdiffusive financial models.
- [Dupret and Hainaut (2023)](https://people.math.ethz.ch/~jdupret/Subdiffusive_SVJ.pdf): subdiffusive stochastic volatility and jumps, with an estimation procedure for illiquid asset returns.
- [Shchestyuk and Tyshchenko (2025)](https://www.vmsta.org/journal/VMSTA/article/314/text): an alternative inverse-Gaussian clock and an Airbnb option-pricing illustration.

These papers have different model and estimation assumptions. They establish that empirical calibration, illiquidity, and applications beyond S&P 500 options are existing research topics. The proposed contribution is a controlled empirical investigation of their generality and sensitivity to observation choices. A focused literature check does not establish priority; a broader comparison of prior methods is still needed before claiming novelty.

## Model scope

The candidate model uses an inverse-stable market-time clock with:

```text
0 < alpha <= 1
```

`alpha = 1` must recover classical BSM under identical inputs. This is not a Hurst-exponent, fractional-Brownian-motion, or rough-volatility project.

The mathematical specification, parameter units and time normalization, volatility/alpha identification, and treatment of American exercise and dividends still require explicit choices. A fitted alpha is not automatically a structural estimate of market waiting times.

## Project status

`collector.py` uses ThetaData exclusively, with hourly stock and option bid/ask snapshots for the hourly-versus-daily pricing study. Normal sessions request the 09:30–15:30 hourly grid and a separate snapshot at 15:55 New York time. The daily comparison snapshot is always five minutes before the actual close, including 12:55 on a 13:00 early close. Daily stock and option EOD reports preserve volume and trade counts; their later report-time quotes are not substituted for the near-close snapshot. Individual trades are not downloaded. Sampled observations cannot reconstruct intervening quote updates or exact trade gaps.

Discovery combines dated quoted/traded contract lists with one bulk OI report per underlying-day, so OI-only contracts remain eligible. The selected expiration, strike, and call/put grid is preserved. Quotes download in batches across all strikes and both rights for each selected expiration: seven shared requests plus two per selected expiration, at most 17 per underlying-day with the five-expiration cap. Extra strikes stay in the raw replies; the saved selected-contract table defines the research sample. Four simultaneous HTTP requests are shared across all workers for Standard, references run in bounded concurrent batches, and the previous artificial request-start delay is disabled by default. This reduces request count, but live download time still needs a representative benchmark.

Every normal run also requests dividends, splits, SOFR and all 11 documented Treasury tenors, and VIX daily, hourly, and near-close prices. Standard index history starts on 2022-01-01: earlier requested VIX sessions are explicitly recorded as subscription coverage gaps without sending inaccessible history requests. Stock, option, index, and rate entitlements are separate; historical rate access must also cover the requested dates. Finer sampled quote intervals remain an explicit configuration choice, with one minute as the minimum supported stock resolution for Standard.

Normal runs include 60 earlier trading sessions of stock hourly/near-close quotes and EOD records by default. `--lookback-sessions` changes this collection buffer; `0` disables it. Rates use the same earlier date; VIX EOD uses that date subject to Standard's index-history limit. Corporate-action requests extend through the study's end date plus the maximum selected DTE, so events before an option's expiration are not cut off at the study boundary. Announcement dates and unknown amounts are preserved; later events are not assumed to have been known at an earlier observation. The lookback does not define a calibration window or guarantee continuous histories for the same option contracts.

Responses download to disk and convert to Parquet in batches. Strict CSV checks reject extra/missing fields and malformed records at every batch boundary. Raw values, conditions, row order, duplicates, and separate clock meanings remain available. Session coverage checks each selected contract inside bulk replies, including absent hourly samples, near-close snapshots, and daily EOD reports. Daily rate/index reports list exchange sessions without a dated observation, including possible differences in publisher holidays. Empty OI and corporate-action responses are not automatically treated as failures. Older parsed caches that lack strict CSV validation require new downloads; original files remain preserved, and pre-refactor caches with verified CSV bytes can be revalidated.

Stock references at 10:30, 13:30, and 15:30 New York time select the research contracts; these times align with the hourly grid. References outside an early-close session are omitted. The combined universe remains an observed list, not proof of complete historical listings. Evaluation samples, regimes, and pricing inputs are left to later work. The current collection supports hourly/daily comparisons; directly studying exact trade gaps would require a separate collection design.

Run `python collector.py --symbols SPY AAPL --start 2018-01-01 --end 2025-12-31 --coverage-only` with Theta Terminal v3 running to check the vendor's available stock and VIX dates before downloading history. `--references-only` collects the reference bundle without stock/option panels. The [current Theta API specification](https://docs.thetadata.us/openapiv3.yaml) documents dividend and split endpoints, but [Theta's history limits](https://docs.thetadata.us/Articles/Data-And-Requests/Making-Requests.html) exclude SPY underlying data before 2020. Adjusted option deliverables, historical symbol mappings, and reference-data vintages remain unverified. The collector reports observed gaps without filling them from another vendor; completed requests do not establish complete research coverage.

No validated BSM or TFBSM pricer, calibration routine, frequency-comparison experiment, or empirical result is included yet. Implementation is being reviewed in pull requests before inclusion on `main`.

## License, citation, and data

Original code is licensed under the [MIT License](LICENSE), copyright 2026 Simon Vu. Citation metadata are in [CITATION.cff](CITATION.cff).

The MIT License covers this repository's original software. It does not grant permission to redistribute ThetaData responses, third-party papers, or other externally owned material. Generated data and results remain local and are ignored by Git.
