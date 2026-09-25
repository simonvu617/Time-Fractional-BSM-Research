# European TFBSM solvers: definitions, citations, and validation

This document defines the model before describing the code. Citations use
`KMP20` for Krzyżanowski, Magdziarz, and Płociniczak [1] and `An24` for An et
al. [2]. Both papers are listed in full, with DOI links, in the references.

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

| alpha | dt | paper maximum error | computed maximum error |
|---:|---:|---:|---:|
| 0.1 | 1/10 | 2.5543e-4 | 2.554339e-4 |
| 0.5 | 1/20 | 1.1428e-4 | 1.142771e-4 |
| 0.9 | 1/40 | 8.7868e-5 | 8.786794e-5 |

KMP20 Table 1 temporal orders are also reproduced:

| alpha | paper order | computed order | theoretical `2-alpha` |
|---:|---:|---:|---:|
| 0.99 | 1.02 | 1.057 | 1.01 |
| 0.70 | 1.32 | 1.359 | 1.30 |
| 0.50 | 1.51 | 1.520 | 1.50 |
| 0.30 | 1.70 | 1.700 | 1.70 |
| 0.10 | 1.85 | 1.850 | 1.90 |

At `alpha=1`, joint space-time refinement from 80 to 320 intervals reduces
the weighted solver's BSM error from `4.65e-2` to `1.61e-3` and the L2
solver's error from `4.67e-2` to `1.60e-3`. For fractional calls, the absolute
cross-solver difference shrinks from `2.73e-5` to `8.03e-6` at `alpha=0.5`
and from `7.92e-5` to `2.73e-5` at `alpha=0.9` over 60--240 intervals.

### Independent subordination benchmark

`validation/subordination.py` does not use either fractional recurrence. It
uses the distributional identity
`S_alpha(t)=t^alpha D_1^(-alpha)`, Kanter's positive-stable representation [3],
and deterministic Gauss--Legendre quadrature. Its clock mean agrees with
`t^alpha/Gamma(1+alpha)` within `1e-5` at quadrature order 256 for
`alpha=0.1, 0.5, 0.9, 0.9999`. The same quadrature reproduces
`E_alpha(-0.05)` within `2.1e-7` across those orders.

Benchmark prices converge independently of either finite-difference method:

| case | order 32 | order 64 | order 128 | order 256 | `|V256-V512|` |
|:---|---:|---:|---:|---:|---:|
| `alpha=.1`, short ATM call | .101133993 | .101139277 | .101140632 | .101140964 | 7.97e-8 |
| `alpha=.3`, long ITM put | .637119071 | .637118225 | .637117954 | .637117875 | 2.16e-8 |
| `alpha=.9999`, ATM put | .103285676 | .103281179 | .103278738 | .103278153 | 6.97e-8 |

The largest `|V128-V256|` in this table is `5.85e-7`, below the tolerances used
to judge the finite-difference methods.

The next table reports absolute error relative to the order-512 benchmark on
`[-4,4]`, with `Nx=Nt=N`. Strikes at `exp(0.5)`, `1`, or `exp(-0.5)` align
with every spatial grid, preventing payoff-grid alignment from obscuring the
refinement trend. All cases use `S=1`. In alpha order, `(K,T,r,sigma)` is
`(1,.25,0,.3)`, `(exp(.5),2,.03,.4)`, `(exp(.5),1,0,.3)`,
`(exp(-.5),1,.03,.35)`, `(exp(-.5),2,.04,.45)`, and `(1,1,.03,.3)`.

| alpha and case | solver | N=80 | N=160 | N=320 | N=640 |
|:---|:---|---:|---:|---:|---:|
| .1, short ATM call | weighted | 3.017e-3 | 7.849e-4 | 2.012e-4 | 5.209e-5 |
|  | L2 | 3.015e-3 | 7.837e-4 | 2.006e-4 | 5.180e-5 |
| .3, long ITM put | weighted | 2.551e-4 | 6.769e-5 | 1.853e-5 | 5.413e-6 |
|  | L2 | 2.497e-4 | 6.515e-5 | 1.729e-5 | 4.809e-6 |
| .5, OTM call | weighted | 2.320e-4 | 8.332e-5 | 3.305e-5 | 1.434e-5 |
|  | L2 | 1.800e-4 | 5.732e-5 | 2.005e-5 | 7.844e-6 |
| .9, OTM put | weighted | 1.850e-4 | 6.258e-5 | 2.373e-5 | 1.015e-5 |
|  | L2 | 1.467e-4 | 4.153e-5 | 1.230e-5 | 3.999e-6 |
| .99, long ITM call | weighted | 3.818e-4 | 1.038e-4 | 3.043e-5 | 1.000e-5 |
|  | L2 | 3.668e-4 | 9.434e-5 | 2.488e-5 | 6.864e-6 |
| .9999, ATM put | weighted | 1.632e-3 | 4.011e-4 | 9.989e-5 | 2.496e-5 |
|  | L2 | 1.634e-3 | 4.015e-4 | 9.999e-5 | 2.498e-5 |

With nonzero `r=0.05`, `S=1`, `K=1.1`, `T=1`, `sigma=0.3`, and `alpha=0.7`,
the benchmark/weighted/L2 call prices are `0.1026890`, `0.1026182`, and
`0.1026366`; put prices are `0.1443118`, `0.1442812`, and `0.1442764`. This
also validates the Mittag--Leffler parity and boundaries away from `r=0`.

### Seven- and fourteen-day maturities

The research-facing short-maturity checks use `S=1`, `r=.03`, the fixed domain
`[-2,2]`, and aligned strikes `1` or `exp(+/-0.0625)`. The table gives absolute
errors against the independent order-512 subordination price as
`N=64/128/256/512`, with `Nx=Nt=N`.

| case | benchmark | weighted errors | L2 errors |
|:---|---:|:---|:---|
| 7d ATM call, `alpha=.1`, `sigma=.2` | .0696660711 | 1.927e-3 / 5.069e-4 / 1.314e-4 / 3.465e-5 | 1.924e-3 / 5.053e-4 / 1.306e-4 / 3.424e-5 |
| 14d slightly ITM put, `alpha=.5`, `sigma=.6` | .1382332752 | 9.018e-4 / 2.549e-4 / 7.780e-5 / 2.643e-5 | 8.955e-4 / 2.519e-4 / 7.641e-5 / 2.576e-5 |
| 7d slightly OTM call, `alpha=.9`, `sigma=1` | .0425209975 | 1.142e-3 / 3.045e-4 / 8.676e-5 / 2.726e-5 | 1.107e-3 / 2.846e-4 / 7.580e-5 / 2.132e-5 |
| 14d slightly OTM put, `alpha=.9999`, `sigma=.6` | .0210735049 | 1.284e-3 / 3.174e-4 / 7.905e-5 / 1.975e-5 | 1.285e-3 / 3.176e-4 / 7.909e-5 / 1.975e-5 |

Every finite-difference error decreases at each refinement. For these cases,
the order-256 to order-512 benchmark change is at most `8.98e-8`, more than
100 times smaller than the finest finite-difference error.

### Near-classical limit

All L2 coefficient arrays through 800 lags are finite at
`alpha=0.99, 0.999, 0.9999, 1`. Prices and complete surfaces contain no
NaNs or infinities. For an ATM call with `r=.03`, `sigma=.3`, and `T=1`, the
order-512 benchmark approaches analytic BSM monotonically:

| alpha | benchmark | distance to BSM | weighted error, N=640 | L2 error, N=640 |
|---:|---:|---:|---:|---:|
| .99 | .1328956641 | 6.258e-5 | 2.790e-5 | 2.604e-5 |
| .999 | .1328395037 | 6.420e-6 | 2.507e-5 | 2.490e-5 |
| .9999 | .1328337366 | 6.526e-7 | 2.478e-5 | 2.479e-5 |
| 1 | .1328330840 | 0 | 2.474e-5 | 2.477e-5 |

Direct high-precision spot checks show relative cancellation in the smallest
late-lag L2 coefficients near one, but the absolute discrepancies remain near
machine precision. The price tests show monotone convergence and no resulting
instability, so the paper's coefficient formulas remain unchanged.

### Separate time, space, and joint refinement

For the `alpha=.5`, `S=K=T=1`, `r=0`, `sigma=.3` call, the table reports error
against the order-512 benchmark. Time refinement fixes `Nx=640`; space
refinement fixes `Nt=640`; joint refinement uses `Nx=Nt=N`.

| mode and solver | N=80 | N=160 | N=320 |
|:---|---:|---:|---:|
| time, weighted | 1.531e-4 | 9.539e-5 | 6.682e-5 |
| time, L2 | 1.458e-4 | 9.196e-5 | 6.517e-5 |
| space, weighted | 2.450e-3 | 6.296e-4 | 1.683e-4 |
| space, L2 | 2.450e-3 | 6.289e-4 | 1.675e-4 |
| joint, weighted | 2.551e-3 | 6.723e-4 | 1.825e-4 |
| joint, L2 | 2.546e-3 | 6.692e-4 | 1.809e-4 |

At these grids, spatial error dominates the residual. All three controlled
sequences decrease for both solvers.

### Domain convergence

The following tests hold `dx=0.025` and `Nt=240` fixed while expanding the
log-price half-width from 2 to 3 to 4. The table reports the absolute change
from half-width 3 to 4; `r=0.03` in every case.

| case | alpha | T | sigma | weighted change | L2 change |
|:---|---:|---:|---:|---:|---:|
| ATM call, short | 0.5 | 0.10 | 0.25 | 6.9e-18 | 6.9e-18 |
| ITM call, long/high vol | 0.9 | 2.00 | 0.60 | 2.0e-8 | 1.5e-8 |
| OTM put, long | 0.5 | 1.50 | 0.30 | 3.0e-11 | 2.0e-11 |
| ITM put, short/high vol | 0.9 | 0.25 | 0.60 | 0.0 | 2.8e-17 |

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

1. G. Krzyżanowski, M. Magdziarz, and Ł. Płociniczak, “A weighted finite
   difference method for subdiffusive Black--Scholes model,” *Computers &
   Mathematics with Applications*, 80(5), 653--670, 2020.
   [doi:10.1016/j.camwa.2020.04.029](https://doi.org/10.1016/j.camwa.2020.04.029);
   [arXiv:1907.00297](https://arxiv.org/abs/1907.00297).
2. X. An, Q. Wang, F. Liu, V. V. Anh, and I. W. Turner, “Parameter estimation
   for time-fractional Black--Scholes equation with S&P 500 index option,”
   *Numerical Algorithms*, 95, 1--30, 2024.
   [doi:10.1007/s11075-023-01563-4](https://doi.org/10.1007/s11075-023-01563-4).
3. M. Kanter, “Stable densities under change of scale and total variation
   inequalities,” *The Annals of Probability*, 3(4), 697--707, 1975.
   [doi:10.1214/aop/1176996309](https://doi.org/10.1214/aop/1176996309).
