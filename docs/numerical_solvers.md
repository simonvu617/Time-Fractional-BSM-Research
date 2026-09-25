# European TFBSM solvers: definitions, citations, and validation

This document defines the model before describing the code. Source keys are
`KMP20`, `An24`, and `Kanter75`; full references and persistent links appear
below. Equation numbers and PDF pages refer to the publisher-version PDFs, not
the journal's running page numbers.

## Paper-to-code citation crosswalk

| Implemented item | Primary-source locator | Code |
|:---|:---|:---|
| Inverse-stable TFBS model, `x=log(S)`, elapsed time | KMP20 equations (4)-(6), PDF pages 5-6; An24 equations (2)-(4), PDF pages 4-5 | `model.py` |
| Weighted L1 coefficients and history | KMP20 equation (7), PDF page 6; weighted recurrence (11), PDF page 7 | `weighted_solver.py` |
| Weighted-method stability and convergence | KMP20 Theorem 3.2, PDF page 10; Theorem 3.3 and optimal weight, PDF pages 15-16 | `weighted_solver.py` |
| L2 startup and fractional coefficients | An24 equations (7)-(9), PDF page 6 | `l2_solver.py` |
| L2 spatial differences and complete recurrence | An24 equations (12)-(17), PDF page 7 | `l2_solver.py` |
| An24 matrix-sign discrepancy | Component equations (14)-(17), PDF page 7, compared with equation (20), PDF page 8 | `l2_solver.py` |
| Independent subordination benchmark | KMP20 Section 2.2, PDF pages 3-5; Kanter75 positive-stable representation | `validation/subordination.py` |

The comments beside each nontrivial recurrence repeat its source locator so the
implementation can be audited without searching this document.

## Shared pricing problem

Let `t` be elapsed time from the payoff toward valuation, measured in years,
and let `x=log(S)`. Both implementations solve

```text
Caputo_D_t^alpha u
  = (sigma^2/2) u_xx + (r-sigma^2/2) u_x - r u,
u(x,0) = payoff(exp(x)),                    0 < alpha <= 1.
```

This is KMP20 equations (4)--(6), pages 5--6, and An24 equations (2)--(4),
pages 4--5. The Caputo derivative is

```text
Caputo_D_t^alpha g(t)
  = 1/Gamma(1-alpha) integral_0^t g'(s)(t-s)^(-alpha) ds,
```

as defined in KMP20 equation (4), page 5, and An24 equation (3), page 5.
`alpha` is dimensionless. An24 introduces `rho` with units
`year^(alpha-1)` and then sets `rho=1` in equation (4), page 5. The code makes
the same normalization, so changing the time unit without rescaling `rho`
would change the model.

KMP20 defines the underlying as geometric Brownian motion evaluated at an
independent inverse alpha-stable subordinator (equation (1), page 2) and uses a
minimal-relative-entropy martingale measure (Proposition 2.1, page 3). An24
introduces its PDE by replacing the ordinary BSM time derivative with a Caputo
derivative and does not supply the same stochastic construction. The shared
interface therefore standardizes the PDE, not an unstated stochastic model.

Both papers use constant `r` and `sigma` and zero dividends. The code does not
add an unsupported dividend term. At `alpha=1`, the Caputo derivative becomes
the ordinary derivative and the model must recover classical BSM.

## Canonical subdiffusive discount and boundaries

KMP20 page 3 gives the subordination identity

```text
V_alpha(t) = E[V_BS(S_alpha(t))],
```

where `S_alpha` is the inverse stable clock. Averaging the operational-time BSM
discount gives `q_alpha(t)=E[exp(-r S_alpha(t))]`. KMP20 page 4 gives the clock
density transform `L_t rho_alpha(s,t)=k^(alpha-1) exp(-s k^alpha)`, hence

```text
L_t q_alpha(k) = k^(alpha-1)/(k^alpha+r),
q_alpha(t)     = E_alpha(-r t^alpha).
```

This produces the coherent inverse-stable relationships

```text
call right = exp(x_max) - K q_alpha(t),    call left = 0,
put left   = K q_alpha(t) - exp(x_min),    put right = 0,
C_alpha-P_alpha = S-K q_alpha(t).
```

They satisfy the printed Caputo PDE because
`Caputo_D_t^alpha q_alpha=-r q_alpha`. The default
`boundary_mode="canonical_subdiffusive"` uses these values. The separate
`boundary_mode="paper_reproduction"` uses `exp(-rt)` to reproduce the papers'
ordinary-discount numerical setup after correcting the evident `x_R` and time
coordinate typos in An24 equations (5)--(6), page 5. At `alpha=1`, both modes
reduce to ordinary BSM boundaries and parity. The dependency-free
Mittag--Leffler evaluator uses its defining series only for
`|r|t^alpha <= 0.75`, which covers the validation and intended market ranges;
it rejects larger arguments rather than risk cancellation error.

## Weighted L1 method

`tfbsm_pricing.weighted_solver` implements KMP20 equations (7)--(11), pages 6--7.
For `dt=T/N`,

```text
b_j = (j+1)^(1-alpha)-j^(1-alpha),
d   = Gamma(2-alpha) dt^alpha.
```

Writing `L_h` for the centered spatial operator, equation (11) becomes

```text
[I-(1-theta)d L_h] U^n
  = sum_(j=0)^(n-2) (b_j-b_(j+1)) U^(n-1-j) + b_(n-1) U^0
    + theta d L_h U^(n-1) + boundary terms.
```

For `n=1`, the history is `U^0`. KMP20 uses `theta=0` for fully implicit and
`theta=1` for fully explicit, the reverse of another common convention. The
default is the stable weight from KMP20 page 16,

```text
theta_hat = (2-2^(1-alpha))/(3-2^(1-alpha)).
```

It becomes `1/2` at `alpha=1`, so the method reduces exactly to
Crank--Nicolson. KMP20 Theorem 3.3, page 15, claims
`O(dt^(2-alpha)+dx^2)` error under its regularity assumptions.

KMP20 Theorem 3.2(i), page 10, labels the scheme unconditionally stable when

```text
1-log2(2-theta/(1-theta)) <= alpha.
```

The parentheses matter. Thus `theta=0` is on the unconditional side for every
`alpha>0`, `theta=0.5` requires `alpha=1`, and `theta_hat(alpha)` lies on the
boundary. The diagnostics implement this exact typeset condition.

## L2 method

`tfbsm_pricing.l2_solver` independently implements An24 equations (7)--(17),
pages 6--7. The first step is the L1 formula in equation (7), with
`phi_1=Gamma(2-alpha)dt^alpha`. Later steps use the quadratic L2 formula in
equations (8)--(9), with `phi_2=Gamma(3-alpha)dt^alpha` and

```text
a_i = (2-alpha)[i^(1-alpha)/2-3(i+1)^(1-alpha)/2]
      -i^(2-alpha)+(i+1)^(2-alpha),
b_i = (4-2alpha)(i+1)^(1-alpha)+2i^(2-alpha)-2(i+1)^(2-alpha),
c_i = (alpha/2-1)[i^(1-alpha)+(i+1)^(1-alpha)]
      -i^(2-alpha)+(i+1)^(2-alpha).
```

The complete history recurrence follows An24 equations (14)--(17), page 7.
At `alpha=1`, its later steps reduce exactly to BDF2 after a backward-Euler
startup. An24 claims order `3-alpha`, but equation (39), page 14, refers to a
first-step transformation without specifying it. The code uses the published
equation (14) startup and does not invent the missing correction.

The component equations (14)--(17) determine the tridiagonal signs. An24
equation (20), page 8, prints a positive upper off-diagonal entry that conflicts
with those component equations and the differential operator, so that sign is
not copied.

## Empirical grid-refinement estimate

The low-level solvers accept an explicit `GridSpec`. `refine_price` jointly
doubles space and time counts on the same finite domain. It evaluates at least
`N`, `2N`, and `4N`, then uses

```text
p_hat = log2(|V_N-V_2N| / |V_2N-V_4N|),
E_hat_4N = |V_4N-V_2N| / (2^p_hat-1).
```

This is an **empirical discretization-error estimate**, not a rigorous bound.
The helper reports convergence only when all observed differences are above a
roundoff floor, decrease strictly, `p_hat` is positive and finite, and the
estimated remaining error is at most the requested tolerance. Otherwise it
returns `converged=False` instead of treating close consecutive prices as
proof of accuracy.

```python
from tfbsm_pricing import GridSpec, refine_price, solve_l2

result = refine_price(
    solve_l2,
    problem,
    GridSpec(-4.0, 4.0, 80, 80),
    tolerance=1e-4,
    max_refinements=3,
)
estimate = result.diagnostics["grid_refinement"]
```

The diagnostics contain every grid level and price, successive differences,
the latest observed order, the estimated remaining discretization error, the
requested tolerance, and the convergence flag. They also record the fixed
log-price domain and state that finite-domain error is excluded. Domain width
is validated separately because

```text
total numerical error
  = finite-domain error + space discretization error + time discretization error.
```

## Independence and sign audit

The solvers share model definitions, boundaries, result types, and a Thomas
factorization. Their Caputo weights, startup rules, and history recurrences are
separate. A shared fractional-time bug therefore cannot make them agree.

For `L=a*d_xx+b*d_x-r`, centered differences give

```text
lower=a/dx^2-b/(2dx),  diag=-2a/dx^2-r,  upper=a/dx^2+b/(2dx).
```

The weighted left side is `I-(1-theta)dL`; the L2 left sides are
`I-phi_1 L` and `beta I-phi_2 L`. Removed boundary entries therefore enter the
right side positively. In the histories, array level `surface[n]` is the
papers' `u^n`; the special L2 startup equations are kept separate from the
general history loop.

## Stability-proof limitation in An24

An24 Theorem 2, pages 11--13, claims unconditional stability. Between
equations (31) and (32), its proof discards a drift cross term that does not
cancel for consecutive time levels. A direct check using
`(x_L,x_R)=(0,pi)`, `alpha=1/2`, `dt=1/100`, `mu=r=1/100`, and
`u^0=sin(30x)` makes the normalized left side of equation (32) equal
`5.130259...`, exceeding the asserted bound `4`. With nonzero drift and two
eigenmodes, the discarded cross term is `-6.7195e-5` at step four, not zero.
This invalidates the printed proof, not the numerical method. Validation below
therefore rests on benchmarks and convergence rather than that theorem.

## Validation results

Run all reproducible checks with:

```powershell
python -m unittest discover -s tests -v
```

The 25 tests cover all 15 An24 Table 1 entries. Representative values are:

| `alpha` | `dt` | Paper error | Computed error |
| :-----: | :--: | ----------: | -------------: |
|   0.10  | 1/10 |  2.5543e-04 |   2.554339e-04 |
|   0.50  | 1/20 |  1.1428e-04 |   1.142771e-04 |
|   0.90  | 1/40 |  8.7868e-05 |   8.786794e-05 |

KMP20 Table 1 temporal orders are also reproduced:

| `alpha` | Paper | Computed | `2-alpha` |
| :-----: | ----: | -------: | --------: |
|   0.99  |  1.02 |    1.057 |      1.01 |
|   0.70  |  1.32 |    1.359 |      1.30 |
|   0.50  |  1.51 |    1.520 |      1.50 |
|   0.30  |  1.70 |    1.700 |      1.70 |
|   0.10  |  1.85 |    1.850 |      1.90 |

At `alpha=1`, joint space-time refinement from 80 to 320 intervals reduces
the weighted solver's BSM error from `4.65e-2` to `1.61e-3` and the L2
solver's error from `4.67e-2` to `1.60e-3`. For fractional calls, the absolute
cross-solver difference shrinks from `2.73e-5` to `8.03e-6` at `alpha=0.5`
and from `7.92e-5` to `2.73e-5` at `alpha=0.9` over 60--240 intervals.

### Independent subordination benchmark

`validation/subordination.py` does not use either fractional recurrence. It
uses the distributional identity
`S_alpha(t)=t^alpha D_1^(-alpha)`, Kanter75's positive-stable representation,
and deterministic Gauss--Legendre quadrature. Its clock mean agrees with
`t^alpha/Gamma(1+alpha)` within `1e-5` at quadrature order 256 for
`alpha=0.1, 0.5, 0.9, 0.9999`. The same quadrature reproduces
`E_alpha(-0.05)` within `2.1e-7` across those orders.

Benchmark prices converge independently of either finite-difference method:

| Case                   |    Order 32 |    Order 64 |   Order 128 |   Order 256 |   Change |
| :--------------------- | ----------: | ----------: | ----------: | ----------: | -------: |
| 0.1000, short ATM call | 0.101133993 | 0.101139277 | 0.101140632 | 0.101140964 | 7.97e-08 |
| 0.3000, long ITM put   | 0.637119071 | 0.637118225 | 0.637117954 | 0.637117875 | 2.16e-08 |
| 0.9999, ATM put        | 0.103285676 | 0.103281179 | 0.103278738 | 0.103278153 | 6.97e-08 |

The largest `|V128-V256|` in this table is `5.85e-7`, below the tolerances used
to judge the finite-difference methods.

The next table reports absolute error relative to the order-512 benchmark on
`[-4,4]`, with `Nx=Nt=N`. Strikes at `exp(0.5)`, `1`, or `exp(-0.5)` align
with every spatial grid, preventing payoff-grid alignment from obscuring the
refinement trend. All cases use `S=1`. In alpha order, `(K,T,r,sigma)` is
`(1,.25,0,.3)`, `(exp(.5),2,.03,.4)`, `(exp(.5),1,0,.3)`,
`(exp(-.5),1,.03,.35)`, `(exp(-.5),2,.04,.45)`, and `(1,1,.03,.3)`.
Cases A-F below follow that order.

| Case | `alpha` | Description    |
| :--: | ------: | :------------- |
|  A   |  0.1000 | Short ATM call |
|  B   |  0.3000 | Long ITM put   |
|  C   |  0.5000 | OTM call       |
|  D   |  0.9000 | OTM put        |
|  E   |  0.9900 | Long ITM call  |
|  F   |  0.9999 | ATM put        |

| Case | Solver   |    `N=80` |   `N=160` |   `N=320` |   `N=640` |
| :--: | :------- | --------: | --------: | --------: | --------: |
|  A   | Weighted | 3.017e-03 | 7.849e-04 | 2.012e-04 | 5.209e-05 |
|  A   | L2       | 3.015e-03 | 7.837e-04 | 2.006e-04 | 5.180e-05 |
|  B   | Weighted | 2.551e-04 | 6.769e-05 | 1.853e-05 | 5.413e-06 |
|  B   | L2       | 2.497e-04 | 6.515e-05 | 1.729e-05 | 4.809e-06 |
|  C   | Weighted | 2.320e-04 | 8.332e-05 | 3.305e-05 | 1.434e-05 |
|  C   | L2       | 1.800e-04 | 5.732e-05 | 2.005e-05 | 7.844e-06 |
|  D   | Weighted | 1.850e-04 | 6.258e-05 | 2.373e-05 | 1.015e-05 |
|  D   | L2       | 1.467e-04 | 4.153e-05 | 1.230e-05 | 3.999e-06 |
|  E   | Weighted | 3.818e-04 | 1.038e-04 | 3.043e-05 | 1.000e-05 |
|  E   | L2       | 3.668e-04 | 9.434e-05 | 2.488e-05 | 6.864e-06 |
|  F   | Weighted | 1.632e-03 | 4.011e-04 | 9.989e-05 | 2.496e-05 |
|  F   | L2       | 1.634e-03 | 4.015e-04 | 9.999e-05 | 2.498e-05 |

With nonzero `r=0.05`, `S=1`, `K=1.1`, `T=1`, `sigma=0.3`, and `alpha=0.7`,
the benchmark/weighted/L2 call prices are `0.1026890`, `0.1026182`, and
`0.1026366`; put prices are `0.1443118`, `0.1442812`, and `0.1442764`. This
also validates the Mittag--Leffler parity and boundaries away from `r=0`.

### Seven- and fourteen-day maturities

The research-facing short-maturity checks use `S=1`, `r=.03`, the fixed domain
`[-2,2]`, and aligned strikes `1` or `exp(+/-0.0625)`. The first table defines
the cases and benchmarks. The second gives absolute error for
`N=64/128/256/512`, with `Nx=Nt=N`.

| Case | DTE | Description       | `alpha` | `sigma` |    Benchmark |
| :--: | --: | :---------------- | ------: | ------: | -----------: |
|  A   |   7 | ATM call          |  0.1000 |    0.20 | 0.0696660711 |
|  B   |  14 | Slightly ITM put  |  0.5000 |    0.60 | 0.1382332752 |
|  C   |   7 | Slightly OTM call |  0.9000 |    1.00 | 0.0425209975 |
|  D   |  14 | Slightly OTM put  |  0.9999 |    0.60 | 0.0210735049 |

| Case | Solver   |    `N=64` |   `N=128` |   `N=256` |   `N=512` |
| :--: | :------- | --------: | --------: | --------: | --------: |
|  A   | Weighted | 1.927e-03 | 5.069e-04 | 1.314e-04 | 3.465e-05 |
|  A   | L2       | 1.924e-03 | 5.053e-04 | 1.306e-04 | 3.424e-05 |
|  B   | Weighted | 9.018e-04 | 2.549e-04 | 7.780e-05 | 2.643e-05 |
|  B   | L2       | 8.955e-04 | 2.519e-04 | 7.641e-05 | 2.576e-05 |
|  C   | Weighted | 1.142e-03 | 3.045e-04 | 8.676e-05 | 2.726e-05 |
|  C   | L2       | 1.107e-03 | 2.846e-04 | 7.580e-05 | 2.132e-05 |
|  D   | Weighted | 1.284e-03 | 3.174e-04 | 7.905e-05 | 1.975e-05 |
|  D   | L2       | 1.285e-03 | 3.176e-04 | 7.909e-05 | 1.975e-05 |

Every finite-difference error decreases at each refinement. For these cases,
the order-256 to order-512 benchmark change is at most `8.98e-8`, more than
100 times smaller than the finest finite-difference error.

### Near-classical limit

All L2 coefficient arrays through 800 lags are finite at
`alpha=0.99, 0.999, 0.9999, 1`. Prices and complete surfaces contain no
NaNs or infinities. For an ATM call with `r=.03`, `sigma=.3`, and `T=1`, the
order-512 benchmark approaches analytic BSM monotonically:

| `alpha` |    Benchmark |   BSM gap | Weighted `N=640` | L2 `N=640` |
| ------: | -----------: | --------: | ---------------: | ---------: |
|  0.9900 | 0.1328956641 | 6.258e-05 |        2.790e-05 |  2.604e-05 |
|  0.9990 | 0.1328395037 | 6.420e-06 |        2.507e-05 |  2.490e-05 |
|  0.9999 | 0.1328337366 | 6.526e-07 |        2.478e-05 |  2.479e-05 |
|  1.0000 | 0.1328330840 | 0.000e+00 |        2.474e-05 |  2.477e-05 |

Direct high-precision spot checks show relative cancellation in the smallest
late-lag L2 coefficients near one, but the absolute discrepancies remain near
machine precision. The price tests show monotone convergence and no resulting
instability, so the paper's coefficient formulas remain unchanged.

### Separate time, space, and joint refinement

For the `alpha=.5`, `S=K=T=1`, `r=0`, `sigma=.3` call, the table reports error
against the order-512 benchmark. Time refinement fixes `Nx=640`; space
refinement fixes `Nt=640`; joint refinement uses `Nx=Nt=N`.

|  Mode | Solver   |    `N=80` |   `N=160` |   `N=320` |
| :---: | :------- | --------: | --------: | --------: |
|  Time | Weighted | 1.531e-04 | 9.539e-05 | 6.682e-05 |
|  Time | L2       | 1.458e-04 | 9.196e-05 | 6.517e-05 |
| Space | Weighted | 2.450e-03 | 6.296e-04 | 1.683e-04 |
| Space | L2       | 2.450e-03 | 6.289e-04 | 1.675e-04 |
| Joint | Weighted | 2.551e-03 | 6.723e-04 | 1.825e-04 |
| Joint | L2       | 2.546e-03 | 6.692e-04 | 1.809e-04 |

At these grids, spatial error dominates the residual. All three controlled
sequences decrease for both solvers.

### Domain convergence

The following tests hold `dx=0.025` and `Nt=240` fixed while expanding the
log-price half-width from 2 to 3 to 4. The table reports the absolute change
from half-width 3 to 4; `r=0.03` in every case.

| Case                    | `alpha` |  `T` | `sigma` | Weighted change | L2 change |
| :---------------------- | ------: | ---: | ------: | --------------: | --------: |
| ATM call, short         |    0.50 | 0.10 |    0.25 |         6.9e-18 |   6.9e-18 |
| ITM call, long/high vol |    0.90 | 2.00 |    0.60 |         2.0e-08 |   1.5e-08 |
| OTM put, long           |    0.50 | 1.50 |    0.30 |         3.0e-11 |   2.0e-11 |
| ITM put, short/high vol |    0.90 | 0.25 |    0.60 |         0.0e+00 |   2.8e-17 |

KMP20 Example 2 is only partially reproduced. The reported ordering by
`theta` agrees, but on the stated `(n,N)=(500,50)` grid the computed errors are
`0.603%`, `0.325%`, and `0.047%` for `theta=0`, `0.25`, and `0.5`, versus the
paper's `1.74%`, `1.12%`, and `0.61%`. The paper does not specify how `S0=1`
is extracted when `log(S0)=0` is not a grid node; this code uses linear
interpolation. Using the two adjacent nodes instead gives `2.67%` and `5.19%`
for `theta=0.5`; reversing the stated space/time counts gives about `1.7%` for
all three theta values. Neither interpretation reproduces the table. KMP20
Table 2 is not used as a spatial benchmark because it says the error is measured
at the prescribed boundary `x=x_max`.

## References

- **KMP20:** G. Krzyżanowski, M. Magdziarz, and Ł. Płociniczak, “A weighted
  finite difference method for subdiffusive Black-Scholes model,” *Computers &
  Mathematics with Applications* **80**(5), 653-670 (2020).
  [Publisher DOI](https://doi.org/10.1016/j.camwa.2020.04.029);
  [open manuscript, arXiv:1907.00297v4](https://arxiv.org/abs/1907.00297v4).

- **An24:** X. An, Q. Wang, F. Liu, V. V. Anh, and I. W. Turner, “Parameter
  estimation for time-fractional Black-Scholes equation with S&P 500 index
  option,” *Numerical Algorithms* **95**, 1-30 (2024). Published online 27 June
  2023. [Publisher DOI and open article](https://doi.org/10.1007/s11075-023-01563-4).

- **Kanter75:** M. Kanter, “Stable densities under change of scale and total
  variation inequalities,” *The Annals of Probability* **3**(4), 697-707
  (1975). [Publisher DOI](https://doi.org/10.1214/aop/1176996309).
