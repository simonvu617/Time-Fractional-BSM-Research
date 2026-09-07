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

`collector.py` saves historical quotes, individual trades with matched quotes, open-interest reports, stock end-of-day records, and the dated contract universe, with request provenance and coverage reports. Quotes default to one-second sampling; tick quotes are optional. Stock references at 10:30, 13:00, and 15:00 New York time select the contracts to download. Yahoo is an optional reference source. Evaluation samples, waiting-time features, regimes, and pricing inputs are left to later work; these collection settings do not select the paper's final comparison frequencies.

No validated BSM or TFBSM pricer, calibration routine, frequency-comparison experiment, or empirical result is included yet. Implementation is being reviewed in pull requests before inclusion on `main`.

## License, citation, and data

Original code is licensed under the [MIT License](LICENSE), copyright 2026 Simon Vu. Citation metadata are in [CITATION.cff](CITATION.cff).

The MIT License covers this repository's original software. It does not grant permission to redistribute ThetaData responses, third-party papers, or other externally owned material. Generated data and results remain local and are ignored by Git.
