# Time-Fractional Black-Scholes Research

This repository is a small empirical-research foundation for testing whether subdiffusive time-fractional Black-Scholes (TFBSM) improves out-of-sample option repricing relative to classical BSM, especially in illiquid or high-waiting-time market regimes.

The answer is unknown. The project must support a positive, negative, or inconclusive result without changing the test after seeing outcomes.

## Model scope

The candidate model uses an inverse-stable market-time clock with:

```text
0 < alpha <= 1
```

`alpha = 1` must recover classical BSM under identical inputs. This is not a Hurst-exponent, fractional-Brownian-motion, or rough-volatility project.

No validated BSM or TFBSM pricer, calibration routine, or empirical result is included yet.

## What is included

The only implemented subsystem is a modular, read-only market-data collector:

```text
Time-Fractional-BSM-Research/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── configs/
│   └── collector.yaml
├── scripts/
│   └── collect_data.py
├── src/
│   └── tfbsm_empirical/
│       ├── __init__.py
│       └── data/
│           ├── client.py
│           ├── config.py
│           ├── cache.py
│           ├── timestamps.py
│           ├── universe.py
│           ├── features.py
│           ├── validation.py
│           ├── writer.py
│           ├── pipeline.py
│           └── cli.py
└── tests/
```

The collector keeps vendor access, configuration, normalization, sampling, storage, validation, and orchestration separate. It writes immutable raw-response receipts and canonical Parquet outputs for one explicitly requested symbol-day.

The collector does **not** price options, select `alpha`, run a backtest, or decide whether TFBSM works. Downstream research should use frozen collector outputs rather than call the vendor again.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

Preview a bounded SPY collection without making a vendor request:

```bash
tfbsm-collect \
  --config configs/collector.yaml \
  --symbol SPY \
  --date 2025-01-03 \
  --dry-run
```

Remove `--dry-run` only when the local ThetaData endpoint is intentionally available. If a replay process is started, stop it after the bounded run.

## Current uncertainty

The automated tests use synthetic data; no live ThetaData collection has been accepted. In particular, the expiration and strike-list requests do not include a historical date. Their point-in-time historical semantics must be established before any multi-year pull.

The starting study window, universe, evaluation times, and sampling grid in `configs/collector.yaml` are inherited research specifications, not validated final choices.

## License, citation, and data

Original code is licensed under the [MIT License](LICENSE), copyright 2026 Simon Vu. Citation metadata are in [CITATION.cff](CITATION.cff).

The MIT License covers this repository's original software. It does not grant permission to redistribute ThetaData responses, third-party papers, or other externally owned material. Generated data and results remain local and are ignored by Git.
