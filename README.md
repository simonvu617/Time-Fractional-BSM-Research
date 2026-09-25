# Time-Fractional Black-Scholes Research

This repository is a small empirical-research foundation for testing whether subdiffusive time-fractional Black-Scholes (TFBSM) improves out-of-sample option repricing relative to classical BSM, especially in illiquid or high-waiting-time market regimes.

The answer is unknown. The project must support a positive, negative, or inconclusive result without changing the test after seeing outcomes.

## Model scope

The candidate model uses an inverse-stable market-time clock with:

```text
0 < alpha <= 1
```

`alpha = 1` must recover classical BSM under identical inputs. This is not a Hurst-exponent, fractional-Brownian-motion, or rough-volatility project.

The `tfbsm_pricing` package contains two independently implemented European
TFBSM finite-difference solvers:

- the weighted L1 method in G. Krzyżanowski, M. Magdziarz, and Ł.
  Płociniczak, [“A weighted finite difference method for subdiffusive
  Black-Scholes model”](https://doi.org/10.1016/j.camwa.2020.04.029),
  *Computers & Mathematics with Applications* 80(5), 653-670 (2020);
- the L2 method in X. An, Q. Wang, F. Liu, V. V. Anh, and I. W. Turner,
  [“Parameter estimation for time-fractional Black-Scholes equation with S&P
  500 index option”](https://doi.org/10.1007/s11075-023-01563-4),
  *Numerical Algorithms* 95, 1-30 (2024).

They share the same model and grid interface but keep their fractional-history
weights and recurrences separate. Neither solver is coupled to the ThetaData
collector.

## Pricing example

```python
from tfbsm_pricing import EuropeanOptionProblem, GridSpec, solve_l2, solve_weighted

problem = EuropeanOptionProblem(
    S0=100, K=100, T=1, r=0.05, sigma=0.20, alpha=0.80, option_type="call"
)
grid = GridSpec(x_min=2, x_max=7, space_steps=300, time_steps=200)
print(solve_weighted(problem, grid).price, solve_l2(problem, grid).price)
```

The paper-to-code formula crosswalk, complete scholarly references, validation
results, and known limitations are in
[`docs/numerical_solvers.md`](docs/numerical_solvers.md).

## Run the checks

```powershell
python -m unittest discover -s tests -v
```

Only NumPy is required.

## Project status

The numerical solvers are being reviewed in a pull request before inclusion on
`main`. Calibration, empirical fitting, American options, trading logic, and
data analysis remain outside this implementation.

## License, citation, and data

Original code is licensed under the [MIT License](LICENSE), copyright 2026 Simon Vu. Citation metadata are in [CITATION.cff](CITATION.cff).

The MIT License covers this repository's original software. It does not grant permission to redistribute ThetaData responses, third-party papers, or other externally owned material. Generated data and results remain local and are ignored by Git.
