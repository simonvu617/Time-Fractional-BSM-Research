# Numerical validation results

These results were generated on 2026-09-24 with Python 3 and NumPy 2.3.5 by
running:

```powershell
python -m unittest discover -s tests -v
python -m validation.run_validation
```

All automated tests passed. The tables distinguish exact paper
reproductions from independent numerical checks.

## An et al. manufactured solution

The exact solution is `(t+1)^2(x^3+x^2+1)` on `[0,1]`, with `r=0.5`,
`sigma^2/2=1`, and the source printed on page 14. These entries reproduce
Table 1 using the published L1 startup and L2 recurrence.

| alpha | dt | space steps | paper max error | computed max error |
|---:|---:|---:|---:|---:|
| 0.1 | 1/10 | 29 | 2.5543e-4 | 2.554339e-4 |
| 0.1 | 1/20 | 78 | 3.5313e-5 | 3.531254e-5 |
| 0.1 | 1/40 | 211 | 4.8264e-6 | 4.826398e-6 |
| 0.5 | 1/10 | 18 | 6.5066e-4 | 6.506561e-4 |
| 0.5 | 1/20 | 43 | 1.1428e-4 | 1.142771e-4 |
| 0.5 | 1/40 | 101 | 2.0731e-5 | 2.073084e-5 |
| 0.9 | 1/10 | 12 | 1.5e-3 | 1.459497e-3 |
| 0.9 | 1/20 | 24 | 3.6555e-4 | 3.655490e-4 |
| 0.9 | 1/40 | 49 | 8.7868e-5 | 8.786794e-5 |

The paper rounds the first `alpha=0.9` entry to two significant digits. The
remaining entries agree to the displayed precision.

## Krzyzanowski et al. temporal order

This uses the parameters and grids stated under Table 1 on page 17, including
the fully implicit method, `dx=0.2`, evaluation at `x=-0.01`, and a reference
time step close to `3.85e-4`.

| alpha | paper order | computed order | theoretical 2-alpha |
|---:|---:|---:|---:|
| 0.99 | 1.02 | 1.057 | 1.01 |
| 0.70 | 1.32 | 1.359 | 1.30 |
| 0.50 | 1.51 | 1.520 | 1.50 |
| 0.30 | 1.70 | 1.700 | 1.70 |
| 0.10 | 1.85 | 1.850 | 1.90 |

The calculated orders reproduce the paper's trend and are close to its printed
values. The spatial-order table is not used as a validation target because it
says the solution is evaluated at `x=x_max`, where the Dirichlet value is
imposed exactly and cannot measure interior spatial error.

## Krzyzanowski et al. Example 2

For the stated `(n,N)=(500,50)` grid, `alpha=0.999`, and BSM benchmark `0.593`:

| theta | paper percent error | computed price | computed percent error |
|---:|---:|---:|---:|
| 0.00 | 1.74% | 0.589424617 | 0.603% |
| 0.25 | 1.12% | 0.591072668 | 0.325% |
| 0.50 | 0.61% | 0.592721380 | 0.047% |

The ranking and convergence toward the Crank--Nicolson weight are reproduced,
but the printed percentages are not. The paper does not specify how `S0=1`
is extracted when `log(S0)=0` is not a node of this grid. This implementation
uses linear interpolation rather than choosing an adjacent node. The result is
therefore recorded as a partial reproduction.

## Classical Black--Scholes limit

At `alpha=1`, the weighted solver becomes Crank--Nicolson and the L2 solver
becomes backward Euler followed by BDF2. The analytic call price is the
independent reference.

| intervals (space,time) | weighted absolute error | L2 absolute error |
|---:|---:|---:|
| (80,80) | 4.652860e-2 | 4.665812e-2 |
| (160,160) | 4.874297e-3 | 4.906624e-3 |
| (320,320) | 1.605036e-3 | 1.596952e-3 |

The remaining error is dominated by the payoff kink, finite domain, and joint
space-time refinement. It decreases for both independent methods.

## Cross-solver convergence

Both solvers use the identical model, domain, payoff, boundary values, and grid.

| alpha | intervals | weighted price | L2 price | absolute difference |
|---:|---:|---:|---:|---:|
| 0.50 | 60 | 0.130040449 | 0.130067760 | 2.73e-5 |
| 0.50 | 120 | 0.134627326 | 0.134642789 | 1.55e-5 |
| 0.50 | 240 | 0.135854524 | 0.135862549 | 8.03e-6 |
| 0.90 | 60 | 0.132989753 | 0.133068919 | 7.92e-5 |
| 0.90 | 120 | 0.136857189 | 0.136905770 | 4.86e-5 |
| 0.90 | 240 | 0.137802155 | 0.137829479 | 2.73e-5 |

The difference shrinks under refinement for both fractional orders. This is
the intended independent-verification result.

## Put--call parity diagnostic

The paper's asserted parity is compared with the numerical difference from the
printed PDE and exponential boundaries.

| alpha | solver | numerical C-P | paper parity | residual |
|---:|:---|---:|---:|---:|
| 0.5 | weighted | -0.052115178 | -0.056868383 | 4.75e-3 |
| 0.5 | L2 | -0.052091611 | -0.056868383 | 4.78e-3 |
| 0.9 | weighted | -0.055333737 | -0.056868383 | 1.53e-3 |
| 0.9 | L2 | -0.055296815 | -0.056868383 | 1.57e-3 |
| 1.0 | weighted | -0.056863319 | -0.056868383 | 5.06e-6 |
| 1.0 | L2 | -0.056863358 | -0.056868383 | 5.03e-6 |

The two solvers agree on the fractional residual, and the residual vanishes to
the grid error at `alpha=1`. This supports the model-level inconsistency
described in `numerical_solvers.md`; it is not evidence of a discrepancy
between the implementations.

## What is validated

- The L2 coefficients, startup, history indexing, forcing, and tridiagonal
  signs reproduce the manufactured-solution table.
- The weighted L1 history reproduces the paper's measured temporal orders.
- Both methods converge to analytic BSM at `alpha=1` and converge toward one
  another for fractional `alpha`.
- Calls and puts are nonnegative on the tested grids, call values are monotone
  in stock price, and all public input validation tests pass.

## What is not validated

- The exact percentages in Krzyzanowski et al. Table 3 are not reproduced;
  the interpolation-based prices are closer to the BSM benchmark.
- The spatial-order Table 2 in that paper is internally ambiguous because it
  evaluates at a prescribed boundary.
- An et al. do not provide the first-step transformation invoked before
  equation (39). The code uses their explicit equation (14), so no unstated
  higher-order startup correction is claimed.
- The unconditional-stability theorem in An et al. is not treated as proven
  because equation (32) drops a nonzero drift contribution.
- The stochastic representation, empirical parameter estimates, calibration,
  and market-data claims are outside this numerical-solver validation.
