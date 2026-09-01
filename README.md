# Time-Fractional Black-Scholes Empirical Research

Research software for testing whether a rigorously implemented subdiffusive time-fractional Black-Scholes model improves **out-of-sample option repricing** relative to classical BSM and stronger empirical benchmarks, especially in liquidity and waiting-time regimes where subdiffusive market time is theoretically relevant.

The empirical answer is unknown. This repository is designed to make a positive result, a negative result, and an underidentified result equally reportable.

## Model lock

The candidate model is the inverse-stable-market-time form of subdiffusive time-fractional Black-Scholes, with fractional order:

```text
0 < alpha <= 1
```

`alpha = 1` must recover the selected classical BSM implementation under identical inputs. This is not a fractional-Brownian-motion, Hurst-exponent, or rough-volatility project.

No accepted BSM or TFBSM pricer is implemented yet. A model does not become accepted merely because a directory or configuration file exists.

## Central empirical questions

1. Is `alpha` empirically below `1` in a way that agrees with independently measured waiting-time behavior?
2. Does TFBSM reduce genuinely out-of-sample repricing error in the same low-liquidity or high-waiting-time regimes?

For observation `i`, the primary improvement metric is:

```text
d_i = absolute_error_i(BSM) - absolute_error_i(TFBSM)
```

A positive value favors TFBSM. The sign and magnitude must be measured; they are not assumed.

## Architecture

```text
Time-Fractional-BSM-Research/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── .gitignore
├── src/
│   └── tfbsm_empirical/
│       ├── __init__.py
│       ├── models/          model specifications and validated pricers
│       ├── data/            collection, provenance, normalization, and panels
│       ├── calibration/     leakage-controlled parameter selection
│       ├── backtesting/     out-of-sample evaluation and replay
│       ├── metrics/         predeclared loss and comparison metrics
│       └── utils/           genuinely cross-cutting utilities
├── scripts/
│   ├── collect_data.py
│   ├── run_experiment.py
│   ├── run_backtest.py
│   └── make_figures.py
├── experiments/
│   └── experiment_001/
│       ├── config.yaml
│       └── README.md
├── notebooks/
│   ├── exploratory/
│   └── analysis/
├── configs/
│   ├── default.yaml
│   ├── model_bsm.yaml
│   └── model_fractional.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   └── processed/
├── results/
│   ├── tables/
│   ├── figures/
│   └── summaries/
├── tests/
├── docs/
│   ├── methodology.md
│   ├── mathematical_model.md
│   ├── reproduction.md
│   └── collector_contract.md
└── paper/
    ├── main.tex
    ├── references.bib
    └── figures/
```

The old monolithic collector, legacy backtests, unvalidated pricing prototype, PDFs, credentials, personal files, generated data, and separate crypto paper-trading experiment were intentionally not imported.

## What is implemented

The current functional vertical slice is the `tfbsm_empirical.data` subsystem:

- validated project-local configuration;
- read-only ThetaData transport;
- immutable request-addressed raw-response storage;
- exchange-session and timestamp normalization;
- declared expiration and strike sampling;
- mechanical quote and waiting-time features;
- required-schema and reconciliation checks;
- content-addressed canonical Parquet outputs;
- one-symbol-day collection manifests.

The repository also contains the locked `alpha` domain and the predeclared absolute-error improvement metric. It does **not** yet contain accepted pricers, calibration, backtests, inference, scientific figures, or empirical findings. The corresponding scripts exit explicitly instead of pretending those stages work.

## Research pipeline

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
  -> tables, figures, and paper
```

Downstream stages must use frozen canonical outputs. They must not contact the vendor again.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Run the local test suite:

```bash
python3 -m unittest discover -s tests -v
```

The tests do not contact ThetaData.

## Experiment 001

The first declared experiment is a bounded SPY collection smoke test, not a model comparison. Inspect its plan without making vendor requests:

```bash
tfbsm-collect \
  --config configs/default.yaml \
  --symbol SPY \
  --date 2025-01-03 \
  --dry-run
```

The live form removes `--dry-run`. Do not scale beyond one audited symbol-day until the manifest, schemas, timestamps, coverage, and missingness have been reviewed. If a local ThetaData replay process is started, stop it after the bounded run.

One material vendor question remains unresolved: the configured expiration and strike-list requests have no historical date parameter. A live smoke test must establish whether they return the correct point-in-time historical chain. If not, the chain-discovery adapter must change before any multi-year pull.

See [Experiment 001](experiments/experiment_001/README.md), [the collector contract](docs/collector_contract.md), and [the reproduction guide](docs/reproduction.md).

## Empirical guardrails

- Lock the mathematical model and numerical solver before testing outcomes.
- Verify the `alpha = 1` BSM limit and numerical convergence.
- Use identical market inputs for BSM and TFBSM comparisons.
- Select `alpha` using training observations only.
- Preserve sample membership, exclusion reasons, quote ages, spreads, quote-change flags, collection failures, model versions, and train/test labels.
- Test entry-IV BSM, no-change, surface-IV BSM, liquidity-adjusted IV BSM, and historical-volatility BSM where supported by audited data.
- Keep repricing evidence separate from the waiting-time evidence needed for a structural mechanism claim.
- Treat American exercise, dividends, timestamp semantics, and vendor data rights as material constraints.

More detail is in [methodology.md](docs/methodology.md) and [mathematical_model.md](docs/mathematical_model.md).

## Data and results

Only directory contracts are committed under `data/` and `results/`. Vendor data and generated outputs remain local and ignored by Git. The [data policy](data/README.md) distinguishes raw, interim, and processed artifacts and makes clear that the repository's software license does not relicense vendor data.

## License and citation

Original software in this repository is available under the [MIT License](LICENSE), copyright 2026 Simon Vu. That license covers the repository's original code; it does not grant rights to ThetaData responses, third-party papers, or other externally owned material.

Academic citation metadata are provided in [CITATION.cff](CITATION.cff). Licensing grants reuse permission; citation records scholarly attribution. They serve different purposes.
