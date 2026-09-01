# Reproduction

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## Local verification

```bash
python3 -m unittest discover -s tests -v
```

The automated tests do not contact ThetaData. They include a one-contract synthetic vertical slice through the client response, immutable raw cache, normalization, universe selection, canonical Parquet writer, reconciliation, and completed manifest.

## Collection plan

```bash
tfbsm-collect \
  --config configs/default.yaml \
  --symbol SPY \
  --date 2025-01-03 \
  --dry-run
```

Removing `--dry-run` makes read-only requests to the configured local ThetaData endpoint. Do not start a broad collection until Experiment 001 establishes historical chain-list semantics and the resulting symbol-day receipt has been audited. Stop the local replay process after the bounded run.

## Reproducing future results

The experiment runner, backtest runner, figure generator, pricers, and calibration code are intentional placeholders. No command currently reproduces a scientific model-comparison result because no accepted result exists. When implemented, every result must identify its experiment configuration, code revision, processed-panel fingerprint, model versions, alpha source, train/test split, and output manifest.
