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

## Payoff and boundaries

On `[x_min,x_max]`, both solvers use

```text
call: left=0,                          right=exp(x_max)-K exp(-rt)
put:  left=K exp(-rt)-exp(x_min),      right=0.
```

The call condition follows KMP20 page 6. An24 equations (5)--(6), page 5,
print `x_R` where dimensional consistency requires `exp(x_R)` and use `T-t`
after defining `t=T-tau`; those are treated as typographical errors. The put
condition retains the finite-left `exp(x_min)` correction.

There is a separate model inconsistency. KMP20 Proposition 2.1, page 3, claims
`C-P=S-K exp(-rt)`, and both papers use the same exponential strike discount
in their boundaries. For `alpha<1`, this function does not solve the printed
Caputo PDE because the Caputo derivative of `exp(-rt)` is not its ordinary
derivative. The homogeneous PDE instead produces a Mittag--Leffler discount.
The implementation retains the papers' exponential boundaries so the two
solvers compare the same published finite-domain problem. The tests expose the
fractional parity residual rather than hiding it. The discrepancy vanishes at
`alpha=1`.

## Weighted L1 method

`tfbsm_pricing.weighted_fd` implements KMP20 equations (7)--(11), pages 6--7.
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

## L2 method

`tfbsm_pricing.l2_fd` independently implements An24 equations (7)--(17),
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

The tests cover all nine An24 Table 1 entries. Representative values are:

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

KMP20 Example 2 is only partially reproduced. The reported ordering by
`theta` agrees, but on the stated `(n,N)=(500,50)` grid the computed errors are
`0.603%`, `0.325%`, and `0.047%` for `theta=0`, `0.25`, and `0.5`, versus the
paper's `1.74%`, `1.12%`, and `0.61%`. The paper does not specify how `S0=1`
is extracted when `log(S0)=0` is not a grid node; this code uses linear
interpolation. KMP20 Table 2 is not used as a spatial benchmark because it says
the error is measured at the prescribed boundary `x=x_max`.

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
