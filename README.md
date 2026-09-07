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

## Project status

Implementation is being reviewed in pull requests before inclusion on `main`.

## License, citation, and data

Original code is licensed under the [MIT License](LICENSE), copyright 2026 Simon Vu. Citation metadata are in [CITATION.cff](CITATION.cff).

The MIT License covers this repository's original software. It does not grant permission to redistribute ThetaData responses, third-party papers, or other externally owned material. Generated data and results remain local and are ignored by Git.
