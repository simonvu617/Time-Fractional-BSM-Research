# European TFBS numerical solvers

This document explains the two European time-fractional Black--Scholes (TFBS)
solvers, the mathematical problem they solve, and the numerical evidence used
to check them. Source keys are `KMP20`, `An24`, and `Kanter75`; full references
and persistent links appear at the end. Equation and page numbers refer to the
publisher-version PDFs.

# What this implementation contains

### Solver 1 -- Weighted L1

File: `tfbsm_pricing/weighted_solver.py`

This solver implements the weighted finite-difference method of Krzyżanowski,
Magdziarz, and Płociniczak (2020), abbreviated `KMP20`. It combines an L1
approximation of the Caputo derivative with a weighted implicit-explicit spatial
step. The paper's `theta` convention is preserved, including its
`alpha`-dependent optimal weight.

### Solver 2 -- L2

File: `tfbsm_pricing/l2_solver.py`

This solver implements the method of An et al. (2024), abbreviated `An24`. It
uses an L1 startup step followed by a quadratic L2 approximation in time, so its
fractional-history coefficients and recurrence are independent of Solver 1.

### Shared model code

File: `tfbsm_pricing/model.py`

This module contains the common problem inputs, `GridSpec`, finite-domain
boundaries, the analytic BSM benchmark, the Mittag--Leffler discount,
tridiagonal factorization, and the `refine_price` helper. It is not a third
solver.

### Independent benchmark

File: `validation/subordination.py`

This is not a third production solver. It exists so the two finite-difference
solvers can be checked against a calculation that does not use either
fractional finite-difference recurrence.

# What are we trying to prove?

Producing a price is not enough. Before using the pricing engine in empirical
work, we want evidence that:

1. the implementations reproduce results from their source papers;
2. `alpha = 1` recovers ordinary Black--Scholes;
3. grid refinement reduces numerical error;
4. the two independent finite-difference methods approach one another;
5. both methods approach an independent subordination benchmark;
6. the benchmark itself is numerically converged;
7. the results are not controlled by an artificially narrow spatial domain;
8. the solvers remain accurate at 7- and 14-day maturities relevant to the
   collected dataset;
9. the L2 solver remains numerically stable when `alpha` is extremely close to
   1; and
10. `refine_price` provides a useful empirical estimate of discretization
    error.

These checks validate different failure modes. Agreement on one check does not
replace the others.

# Shared mathematical problem

The input `S0` is the current underlying price, `K` is the strike, `T` is the
time to maturity in years, `r` is the constant risk-free rate, `sigma` is the
constant volatility, and `alpha` is the dimensionless fractional parameter.
Both papers assume zero dividends.

The solvers use elapsed time `t`, measured forward from the payoff at `t=0`
toward the requested value at `t=T`, and transform asset price with

```math
x=\log S.
```

They solve the same canonical transformed PDE:

```math
{}^C_0D_t^\alpha u(x,t)
=
\frac{\sigma^2}{2}u_{xx}(x,t)
+
\left(r-\frac{\sigma^2}{2}\right)u_x(x,t)
-r\,u(x,t),
\qquad 0<\alpha\le1,
```

with initial condition

```math
u(x,0)=\operatorname{payoff}(e^x).
```

This is KMP20 equations (4)--(6), PDF pages 5--6, and An24 equations
(2)--(4), PDF pages 4--5. The Caputo derivative is defined in KMP20
equation (4), PDF page 5, and An24 equation (3), PDF page 5:

```math
{}^C_0D_t^\alpha g(t)
=
\frac{1}{\Gamma(1-\alpha)}
\int_0^t
g'(s)(t-s)^{-\alpha}\,ds,
\qquad 0<\alpha<1.
```

At `alpha = 1`, it becomes the ordinary first derivative and the model must
recover classical BSM. Smaller `alpha` gives stronger subdiffusive
time-memory behavior. That statement describes the model; it is not an
empirical interpretation of fitted market behavior.

An24 introduces a scale parameter `rho` with units
$\text{year}^{\alpha-1}$, then sets `rho = 1` in equation (4), PDF page 5.
The code uses the same normalization. Changing the unit of time without
rescaling `rho` would therefore change the model.

## Inverse-stable interpretation

KMP20 defines the underlying as geometric Brownian motion evaluated at an
independent inverse `alpha`-stable clock (equation (1), PDF page 2) under the
minimal-relative-entropy martingale measure of Proposition 2.1, PDF page 3.
An24 starts from the same TFBS PDE but does not provide that stochastic
construction. The shared interface standardizes the PDE, without attributing
an unstated stochastic model to An24.

The inverse-stable representation is

```math
V_\alpha(T)
=
\mathbb{E}\!\left[
V_{\mathrm{BS}}\!\left(S_\alpha(T)\right)
\right],
```

where $S_\alpha(T)$ is random operational time. This identity later provides
the independent benchmark.

# Boundary conditions and fractional discounting

The inverse-stable clock changes discounting. Define

```math
q_\alpha(t)
=
\mathbb{E}\!\left[e^{-rS_\alpha(t)}\right]
=
E_\alpha(-rt^\alpha),
```

where $E_\alpha$ is the Mittag--Leffler function. KMP20 Section 2.2, PDF
pages 3--5, gives the subordination identity and the clock-density transform
from which this relationship follows. In Laplace space,

```math
\mathcal{L}_t\{q_\alpha\}(k)
=
\frac{k^{\alpha-1}}{k^\alpha+r}.
```

In elapsed time, the discount satisfies

```math
{}^C_0D_t^\alpha q_\alpha(t)=-r q_\alpha(t).
```

The default `boundary_mode="canonical_subdiffusive"` uses

```math
\begin{aligned}
C(x_{\min},t)&=0,
&
C(x_{\max},t)&=e^{x_{\max}}-Kq_\alpha(t),\\
P(x_{\min},t)&=Kq_\alpha(t)-e^{x_{\min}},
&
P(x_{\max},t)&=0.
\end{aligned}
```

The corresponding fractional put--call parity is

```math
C_\alpha-P_\alpha=S-Kq_\alpha(t).
```

The alternative `boundary_mode="paper_reproduction"` uses the ordinary
discount $e^{-rt}$. It exists only to reproduce the papers' published
numerical setups, after correcting the evident `x_R` and time-coordinate typos
in An24 equations (5)--(6), PDF page 5. At `alpha = 1`, both boundary modes
reduce to ordinary BSM discounting and parity.

The dependency-free Mittag--Leffler evaluator uses its defining series for

```math
| r | t^\alpha\le0.75. |
```

That range covers the validation cases and intended market inputs. Larger
arguments are rejected because the direct series is not a reliable general
large-argument algorithm.

# Why implement two solvers?

The two methods approximate the same PDE with different fractional time
discretizations. If both approach the same price as their grids are refined,
that is stronger evidence than testing only one recurrence.

However,

```math
\text{solver agreement}\ne\text{proof of correctness}.
```

The methods share the same model definition and could share a model-level
mistake. The independent subordination benchmark is therefore also necessary.

# Solver 1: Weighted L1 method

The implementation follows KMP20 equations (7)--(11), PDF pages 6--7.

## 1. Spatial discretization

On a uniform log-price grid, centered differences approximate the spatial
operator

```math
L_hu
=
\frac{\sigma^2}{2}u_{xx}
+
\left(r-\frac{\sigma^2}{2}\right)u_x
-r u.
```

Write `a = sigma^2 / 2`, `b = r - a`, and `dx` for the spatial step. The
centered operator has coefficients

```math
\text{lower}=\frac{a}{\Delta x^2}-\frac{b}{2\Delta x},
\qquad
\text{diag}=-\frac{2a}{\Delta x^2}-r,
\qquad
\text{upper}=\frac{a}{\Delta x^2}+\frac{b}{2\Delta x}.
```

This produces a tridiagonal operator: each interior node depends only on its
left neighbor, itself, and its right neighbor.

## 2. Caputo L1 approximation

For `dt = T / Nt`, define

```math
d=\Gamma(2-\alpha)\,\Delta t^\alpha
```

and the L1 weights

```math
b_j
=
(j+1)^{1-\alpha}-j^{1-\alpha}.
```

At time level $n$, the L1 approximation can be written as

```math
{}^C_0D_t^\alpha U^n
\approx
\frac{1}{d}
\left[
U^n
-\sum_{j=0}^{n-2}(b_j-b_{j+1})U^{n-1-j}
-b_{n-1}U^0
\right].
```

The first term is the unknown current solution. The sum is the fractional
history, ordered from the newest earlier level back toward the payoff, and the
last term retains the initial payoff contribution.

## 3. Meaning of `theta`

KMP20 blends the spatial operator at the previous and current time levels:

```math
\left[I-(1-\theta)dL_h\right]U^n
=
\sum_{j=0}^{n-2}(b_j-b_{j+1})U^{n-1-j}
+b_{n-1}U^0
+\theta dL_hU^{n-1}
+\text{boundary terms}.
```

The matrix on the left is the implicit part. The $\theta dL_hU^{n-1}$ term
is the explicit contribution, while the remaining terms carry the entire
fractional history.

KMP20 uses `theta = 0` for fully implicit and `theta = 1` for fully explicit,
which is the reverse of another common convention. The default is the
paper's optimal stable weight after Theorem 3.3, PDF page 16,

```math
\widehat{\theta}
=
\frac{2-2^{1-\alpha}}{3-2^{1-\alpha}}.
```

At `alpha = 1`, $\widehat{\theta}=1/2$, and the method reduces exactly to
Crank--Nicolson.

## 4. Matrix solve and history cost

The left-hand tridiagonal matrix is factored once and reused at every time
step. The right-hand side must still combine all prior time levels because a
Caputo derivative has memory. Unlike ordinary BSM time stepping, the current
fractional solution depends on the full earlier path.

KMP20 Theorem 3.3, PDF page 15, claims

```math
O\!\left(\Delta t^{\,2-\alpha}+\Delta x^2\right)
```

under its regularity assumptions. Theorem 3.2(i), PDF page 10, gives the
unconditional-stability region

```math
1-\log_2\!\left(2-\frac{\theta}{1-\theta}\right)\le\alpha.
```

The diagnostics implement this exact parenthesized condition. Consequently,
`theta = 0` is on the unconditional side for every `alpha > 0`, `theta = 0.50`
requires `alpha = 1`, and the optimal weight lies on the boundary of the stated
region.

# Solver 2: L2 method

The implementation follows An24 equations (7)--(17), PDF pages 6--7.

## 1. L1 startup

A quadratic L2 history formula needs more than one earlier level. The first
step therefore uses the paper's L1 formula with

```math
\phi_1=\Gamma(2-\alpha)\,\Delta t^\alpha.
```

This startup is kept separate in the code rather than being hidden inside the
later recurrence.

## 2. Quadratic L2 history approximation

For later steps, An24 uses

```math
\phi_2=\Gamma(3-\alpha)\,\Delta t^\alpha
```

and three coefficient sequences:

```math
\begin{aligned}
a_i
&=
(2-\alpha)
\left[
\frac{i^{1-\alpha}}{2}
-\frac{3(i+1)^{1-\alpha}}{2}
\right]
-i^{2-\alpha}
+(i+1)^{2-\alpha},\\
b_i
&=
(4-2\alpha)(i+1)^{1-\alpha}
+2i^{2-\alpha}
-2(i+1)^{2-\alpha},\\
c_i
&=
\left(\frac{\alpha}{2}-1\right)
\left[i^{1-\alpha}+(i+1)^{1-\alpha}\right]
-i^{2-\alpha}
+(i+1)^{2-\alpha}.
\end{aligned}
```

These are An24 equations (8)--(9), PDF page 6. Their combinations multiply
earlier solution levels in equations (15)--(17), PDF page 7. Separate formulas
are required for the second and third steps before the general history pattern
is available.

## 3. Spatial solve

The method uses the same centered approximation of the canonical spatial
operator, but its time coefficients, startup rule, and history recurrence are
independent of the weighted solver. Each new level solves a tridiagonal system
whose right-hand side contains the complete L2 history and current boundary
values. In compact form, the weighted left matrix is
$I-(1-\theta)dL_h$, while the two L2 left matrices are

```math
I-\phi_1L_h
\qquad\text{and}\qquad
\beta I-\phi_2L_h.
```

Removing boundary columns from these matrices moves their contributions to the
right-hand side with positive signs. Array level `surface[n]` is the papers'
$u^n$, and the special L2 startup steps remain separate from the general
history loop.

At `alpha = 1`, the startup becomes backward Euler and the later steps become
BDF2. An24 claims temporal order

```math
O\!\left(\Delta t^{\,3-\alpha}\right)
```

for sufficiently smooth solutions. A European option payoff has a kink at the
strike, so this smooth-solution order should not be treated automatically as
the observed production-pricing order.

An24 equation (39), PDF page 14, refers to a first-step transformation without
specifying it. The implementation uses the published equation (14) startup and
does not invent the missing correction. Equation (20), PDF page 8, also prints
an upper off-diagonal sign that conflicts with component equations (14)--(17)
and the differential operator; the implementation follows the component
equations.

# Weighted L1 vs L2

| Property                          | Weighted L1             | L2                       |
| :-------------------------------- | :---------------------- | :----------------------- |
| Source                            | KMP20                   | An24                     |
| Time approximation                | L1                      | L2                       |
| Claimed temporal order            | $2-\alpha$              | $3-\alpha$, if smooth    |
| `alpha = 1` limit                 | Crank--Nicolson         | BE startup + BDF2        |
| Independent benchmark convergence | Yes                     | Yes                      |
| 7/14 DTE validated                | Yes                     | Yes                      |
| Near `alpha = 1` tested           | Yes                     | Yes                      |
| Runtime benchmark                 | Not yet                 | Not yet                  |

The L2 method has smaller errors on some tested grids, but the evidence does
not establish a universal winner. Accuracy depends on `alpha`, payoff
regularity, grid balance, domain width, and the requested tolerance.

# Independent subordination benchmark

The independent reference uses

```math
V_\alpha(T)
=
\mathbb{E}
\left[
V_{\mathrm{BS}}\!\left(S_\alpha(T)\right)
\right].
```

Here $S_\alpha(T)$ is random operational time generated by the inverse-stable
clock. Conditional on one operational-time realization, ordinary analytic BSM
can be evaluated. Averaging those conditional prices gives a TFBS price without
using either finite-difference history recurrence.

`validation/subordination.py` evaluates this expectation with the
distributional identity

```math
S_\alpha(t)\overset{d}{=}t^\alpha D_1^{-\alpha},
```

Kanter75's positive-stable representation, and deterministic Gauss--Legendre
quadrature. Increasing the quadrature order provides a direct convergence
check before the value is used as a reference.

> [!NOTE]
> The subordination benchmark is independent of both finite-difference history
> recurrences.

# Empirical grid-refinement estimate

The low-level solvers accept an explicit `GridSpec`. Starting from a grid with
joint resolution `N`, `refine_price` computes

```math
V_N,\qquad V_{2N},\qquad V_{4N},
```

on the same finite log-price domain. It estimates the observed order with

```math
\widehat p
=
\log_2
\left(
\frac{|V_N-V_{2N}|}
{|V_{2N}-V_{4N}|}
\right)
```

and the remaining error at the finest level with

```math
\widehat E_{4N}
=
\frac{|V_{4N}-V_{2N}|}{2^{\widehat p}-1}.
```

In plain language, the helper asks whether successive grid refinements are
changing the price by less and less, estimates how quickly those changes are
shrinking, and extrapolates the remaining discretization error. It reports
`converged=False` if the differences do not decrease regularly, the observed
order is not positive and finite, or the estimate exceeds the requested
tolerance.

> [!IMPORTANT]
> `refine_price` estimates space/time discretization error on the chosen finite
> domain. It does not include finite-domain truncation error and is not a
> rigorous exact error bound.

The distinction can be summarized as

```math
\text{total numerical error}
=
\text{finite-domain error}
+
\text{space discretization error}
+
\text{time discretization error}.
```

# Validation

Run all reproducible checks with:

~~~powershell
python -m unittest discover -s tests -v
~~~

The suite currently contains 25 tests.

`Nx` is the number of spatial intervals in log-price, and `Nt` is the number
of elapsed-time intervals. When a table uses only `N`, it means

```math
N_x=N_t=N.
```

Increasing these counts costs more computation but generally reduces
discretization error. Grid resolution is distinct from domain width
$[x_{\min},x_{\max}]$: refining `Nx` on a fixed domain makes the mesh finer,
while expanding the domain tests whether artificial boundaries affect the
price.

Unless stated otherwise, convergence tables report absolute error,

```math
\left|V_{\mathrm{solver}}-V_{\mathrm{benchmark}}\right|.
```

Moving left to right means refining the grid. Falling values indicate
convergence toward the stated benchmark.

## Validation summary

| Validation                 | What it checks                         | Result                        |
| :------------------------- | :------------------------------------- | :---------------------------- |
| An24 Table 1               | L2 coefficients and recurrence         | Passed all 15 cases           |
| KMP20 temporal convergence | Weighted L1 recurrence                 | Reproduced                    |
| `alpha = 1`                | Classical BSM limit                    | Passed                        |
| Solver vs solver           | Independent FD implementations         | Difference decreases          |
| Subordination benchmark    | Fractional-price correctness           | Both converge to benchmark    |
| Benchmark quadrature       | Independent-reference accuracy         | Stabilizes with order         |
| Broad `alpha` range        | Behavior from 0.10 to 0.9999           | Passed                        |
| 7/14 DTE                   | Short-maturity behavior                | Passed                        |
| Near `alpha = 1`           | Numerical cancellation and stability   | No pricing instability seen   |
| Domain expansion           | Boundary truncation                    | Prices stabilize              |
| `refine_price`             | Practical discretization estimate      | Implemented and validated     |

> [!IMPORTANT]
> Agreement between the two PDE solvers is evidence of correctness. Their
> convergence toward the independent benchmark is the stronger check.

## An24 manufactured-solution reproduction

### What is being tested?

The test reproduces all 15 entries in An24 Table 1 using the paper's
manufactured solution.

### Why does it matter?

A manufactured solution checks the L2 coefficients, startup, history
recurrence, spatial operator, and boundary/source handling against a known
answer.

### How was it tested?

The published problem and grids were supplied to `solve_l2_pde`. The automated
suite checks all 15 rows; representative entries appear below.

### Results

| `alpha` | `dt` | Paper error | Computed error |
| :-----: | :--: | ----------: | -------------: |
| 0.10    | 1/10 | 2.5543e-04  | 2.554339e-04   |
| 0.50    | 1/20 | 1.1428e-04  | 1.142771e-04   |
| 0.90    | 1/40 | 8.7868e-05  | 8.786794e-05   |

### Interpretation

The computed errors reproduce the published values at the precision shown,
supporting the implemented L2 recurrence and coefficient signs.

## KMP20 temporal convergence

### What is being tested?

This test checks whether the weighted method reproduces the temporal orders
reported in KMP20 Table 1.

### Why does it matter?

It targets the weighted L1 history recurrence and the `alpha`-dependent
convergence behavior directly.

### How was it tested?

The paper's temporal-convergence problem was repeated while controlling the
spatial error. The last column is the theoretical smooth-solution order.

### Results

| `alpha` | Paper | Computed | `2-alpha` |
| :-----: | ----: | -------: | --------: |
| 0.99    | 1.02  | 1.057    | 1.01      |
| 0.70    | 1.32  | 1.359    | 1.30      |
| 0.50    | 1.51  | 1.520    | 1.50      |
| 0.30    | 1.70  | 1.700    | 1.70      |
| 0.10    | 1.85  | 1.850    | 1.90      |

### Interpretation

The observed orders reproduce the paper's trend and remain close to
$2-\alpha$, with the same finite-grid departures from the asymptotic value.

## Classical BSM limit

### What is being tested?

This test asks whether both solvers recover analytic Black--Scholes pricing at
`alpha = 1`.

### Why does it matter?

Failure here would reveal an incorrect limiting recurrence, discount,
boundary, or spatial operator.

### How was it tested?

Both space and time were refined jointly from 80 to 320 intervals, and each
price was compared with the analytic zero-dividend BSM value.

### Results

| Solver   | `N=80` error | `N=320` error |
| :------- | -----------: | ------------: |
| Weighted | 4.65e-02     | 1.61e-03      |
| L2       | 4.67e-02     | 1.60e-03      |

### Interpretation

Both errors fall by roughly an order of magnitude and approach the same
analytic BSM limit. This also checks the Crank--Nicolson and BE/BDF2 limiting
forms described above.

## Cross-solver convergence

### What is being tested?

This test measures the absolute difference between the two finite-difference
prices as their common grid is refined.

### Why does it matter?

The two schemes have separate fractional weights and time recurrences. A
shrinking difference is evidence that they approach the same PDE solution.

### How was it tested?

Fractional ATM calls were solved on common 60-, 120-, and 240-interval grids.
The table shows the endpoint differences.

### Results

| `alpha` | `N=60` difference | `N=240` difference |
| :-----: | ----------------: | -----------------: |
| 0.50    | 2.73e-05          | 8.03e-06           |
| 0.90    | 7.92e-05          | 2.73e-05           |

### Interpretation

The solver-to-solver difference decreases in both cases. This is useful
cross-validation, but it does not by itself exclude a shared model-level error.

## Benchmark quadrature convergence

### What is being tested?

This test checks whether the independent subordination calculation stabilizes
as the deterministic quadrature order increases.

### Why does it matter?

A benchmark must be more accurate than the finite-difference errors it is used
to judge.

### How was it tested?

Orders 32 through 512 were evaluated. `Change` is
$\left|V_{256}-V_{512}\right|$; the displayed price columns stop at order
256 to keep the table narrow.

### Results

| Case                   | Order 32    | Order 64    | Order 128   | Order 256   | Change   |
| :--------------------- | ----------: | ----------: | ----------: | ----------: | -------: |
| 0.1000, short ATM call | 0.101133993 | 0.101139277 | 0.101140632 | 0.101140964 | 7.97e-08 |
| 0.3000, long ITM put   | 0.637119071 | 0.637118225 | 0.637117954 | 0.637117875 | 2.16e-08 |
| 0.9999, ATM put        | 0.103285676 | 0.103281179 | 0.103278738 | 0.103278153 | 6.97e-08 |

The largest $\left|V_{128}-V_{256}\right|$ is `5.85e-07`. At order 256,
the clock mean agrees with

```math
\frac{t^\alpha}{\Gamma(1+\alpha)}
```

within `1.00e-05` for `alpha = 0.10`, `0.50`, `0.90`, and `0.9999`. The same
quadrature reproduces $E_\alpha(-0.05)$ within `2.10e-07`.

### Interpretation

Successive benchmark changes fall below the tolerances used for the PDE
solvers. The reference error is therefore small relative to the
finite-difference errors in the following comparisons.

## Solver convergence to subordination

### What is being tested?

This test compares both PDE solvers with the independent benchmark when the
interest rate is nonzero.

### Why does it matter?

It checks the canonical Mittag--Leffler boundaries and put--call parity as well
as fractional pricing away from the simpler `r = 0` cases.

### How was it tested?

The problem uses `S0 = 1.00`, `K = 1.10`, `T = 1.00`, `r = 0.05`,
`sigma = 0.30`, and `alpha = 0.70`. Both calls and puts were priced.

### Results

| Option | Benchmark | Weighted  | L2        |
| :----: | --------: | --------: | --------: |
| Call   | 0.1026890 | 0.1026182 | 0.1026366 |
| Put    | 0.1443118 | 0.1442812 | 0.1442764 |

### Interpretation

Both solvers are close to the independent values for calls and puts. Their
call-minus-put differences also satisfy the fractional parity relation using
the Mittag--Leffler discount.

## Broad `alpha` range

### What is being tested?

This is the main benchmark comparison across
`alpha = 0.10`, `0.30`, `0.50`, `0.90`, `0.99`, and `0.9999`.

### Why does it matter?

It checks that convergence is not confined to one convenient fractional
parameter, maturity, moneyness, rate, or volatility.

### How was it tested?

All cases use `S0 = 1.00`, the domain `[-4, 4]`, and the order-512
subordination benchmark. Strikes align with every spatial grid so payoff-grid
alignment does not obscure the refinement trend.

| Case | `alpha` | `K`          | `T`  | `r`  | `sigma` | Option   |
| :--: | ------: | :----------- | ---: | ---: | ------: | :------- |
| A    | 0.1000  | 1.0000       | 0.25 | 0.00 | 0.30    | ATM call |
| B    | 0.3000  | $e^{0.5}$    | 2.00 | 0.03 | 0.40    | ITM put  |
| C    | 0.5000  | $e^{0.5}$    | 1.00 | 0.00 | 0.30    | OTM call |
| D    | 0.9000  | $e^{-0.5}$   | 1.00 | 0.03 | 0.35    | OTM put  |
| E    | 0.9900  | $e^{-0.5}$   | 2.00 | 0.04 | 0.45    | ITM call |
| F    | 0.9999  | 1.0000       | 1.00 | 0.03 | 0.30    | ATM put  |

### Results

All entries are absolute errors relative to the independent benchmark.

| Case | Solver   | `N=80`    | `N=160`   | `N=320`   | `N=640`   |
| :--: | :------- | --------: | --------: | --------: | --------: |
| A    | Weighted | 3.017e-03 | 7.849e-04 | 2.012e-04 | 5.209e-05 |
| A    | L2       | 3.015e-03 | 7.837e-04 | 2.006e-04 | 5.180e-05 |
| B    | Weighted | 2.551e-04 | 6.769e-05 | 1.853e-05 | 5.413e-06 |
| B    | L2       | 2.497e-04 | 6.515e-05 | 1.729e-05 | 4.809e-06 |
| C    | Weighted | 2.320e-04 | 8.332e-05 | 3.305e-05 | 1.434e-05 |
| C    | L2       | 1.800e-04 | 5.732e-05 | 2.005e-05 | 7.844e-06 |
| D    | Weighted | 1.850e-04 | 6.258e-05 | 2.373e-05 | 1.015e-05 |
| D    | L2       | 1.467e-04 | 4.153e-05 | 1.230e-05 | 3.999e-06 |
| E    | Weighted | 3.818e-04 | 1.038e-04 | 3.043e-05 | 1.000e-05 |
| E    | L2       | 3.668e-04 | 9.434e-05 | 2.488e-05 | 6.864e-06 |
| F    | Weighted | 1.632e-03 | 4.011e-04 | 9.989e-05 | 2.496e-05 |
| F    | L2       | 1.634e-03 | 4.015e-04 | 9.999e-05 | 2.498e-05 |

### Interpretation

Every row decreases under every refinement, and each finest-grid error is less
than one tenth of its coarsest-grid error. Both solvers converge to the
independent reference throughout the tested range.

## Near-`alpha = 1` behavior

### What is being tested?

This test looks for overflow, non-finite coefficients, or price instability as
the fractional model approaches ordinary BSM.

### Why does it matter?

The L2 coefficient formulas contain differences between nearly equal terms
near `alpha = 1`, so cancellation is a plausible numerical concern.

### How was it tested?

All L2 coefficient arrays through 800 lags were checked at
`alpha = 0.99`, `0.999`, `0.9999`, and `1.00`. Prices and full surfaces were
also checked for finite values. The table uses an ATM call with `r = 0.03`,
`sigma = 0.30`, and `T = 1.00`.

### Results

| `alpha` | Benchmark    | BSM gap   | Weighted `N=640` | L2 `N=640` |
| ------: | -----------: | --------: | ---------------: | ---------: |
| 0.9900  | 0.1328956641 | 6.258e-05 | 2.790e-05        | 2.604e-05  |
| 0.9990  | 0.1328395037 | 6.420e-06 | 2.507e-05        | 2.490e-05  |
| 0.9999  | 0.1328337366 | 6.526e-07 | 2.478e-05        | 2.479e-05  |
| 1.0000  | 0.1328330840 | 0.000e+00 | 2.474e-05        | 2.477e-05  |

### Interpretation

The benchmark approaches analytic BSM monotonically. High-precision spot
checks detect relative cancellation in the smallest late-lag L2 coefficients,
but the absolute discrepancies remain near machine precision, and no pricing
instability was observed.

## Seven- and fourteen-day maturities

### What is being tested?

The collector begins with 7- and 14-calendar-day maturity targets, so these
tests check the numerical solvers in the shortest-maturity region expected in
the dataset.

### Why does it matter?

Short maturity concentrates payoff curvature near the strike and can make
spatial resolution more demanding.

### How was it tested?

All cases use `S0 = 1.00`, `r = 0.03`, the fixed domain `[-2, 2]`, and the
order-512 benchmark. Strikes are `1.00`, $e^{+0.0625}$, or $e^{-0.0625}$, deliberately
aligned with every tested grid; off-grid strike behavior is therefore not
isolated by this particular test.

| Case | DTE | Description       | `alpha` | `sigma` | Benchmark    |
| :--: | --: | :---------------- | ------: | ------: | -----------: |
| A    | 7   | ATM call          | 0.1000  | 0.20    | 0.0696660711 |
| B    | 14  | Slightly ITM put  | 0.5000  | 0.60    | 0.1382332752 |
| C    | 7   | Slightly OTM call | 0.9000  | 1.00    | 0.0425209975 |
| D    | 14  | Slightly OTM put  | 0.9999  | 0.60    | 0.0210735049 |

### Results

All entries are absolute errors relative to the independent benchmark.

| Case | Solver   | `N=64`    | `N=128`   | `N=256`   | `N=512`   |
| :--: | :------- | --------: | --------: | --------: | --------: |
| A    | Weighted | 1.927e-03 | 5.069e-04 | 1.314e-04 | 3.465e-05 |
| A    | L2       | 1.924e-03 | 5.053e-04 | 1.306e-04 | 3.424e-05 |
| B    | Weighted | 9.018e-04 | 2.549e-04 | 7.780e-05 | 2.643e-05 |
| B    | L2       | 8.955e-04 | 2.519e-04 | 7.641e-05 | 2.576e-05 |
| C    | Weighted | 1.142e-03 | 3.045e-04 | 8.676e-05 | 2.726e-05 |
| C    | L2       | 1.107e-03 | 2.846e-04 | 7.580e-05 | 2.132e-05 |
| D    | Weighted | 1.284e-03 | 3.174e-04 | 7.905e-05 | 1.975e-05 |
| D    | L2       | 1.285e-03 | 3.176e-04 | 7.909e-05 | 1.975e-05 |

### Interpretation

Every error decreases at every refinement. The order-256 to order-512
benchmark change is at most `8.98e-08`, more than 100 times smaller than the
finest finite-difference error.

## Separate time, space, and joint refinement

### What is being tested?

This test separates temporal and spatial refinement instead of changing both
at once.

### Why does it matter?

Joint refinement can hide which discretization controls the remaining error.

### How was it tested?

For the `alpha = 0.50`, `S0 = K = T = 1.00`, `r = 0.00`,
`sigma = 0.30` call, time refinement fixes `Nx = 640`, space refinement fixes
`Nt = 640`, and joint refinement uses `Nx = Nt = N`. Errors are relative to
the order-512 benchmark.

### Results

| Mode  | Solver   | `N=80`    | `N=160`   | `N=320`   |
| :---: | :------- | --------: | --------: | --------: |
| Time  | Weighted | 1.531e-04 | 9.539e-05 | 6.682e-05 |
| Time  | L2       | 1.458e-04 | 9.196e-05 | 6.517e-05 |
| Space | Weighted | 2.450e-03 | 6.296e-04 | 1.683e-04 |
| Space | L2       | 2.450e-03 | 6.289e-04 | 1.675e-04 |
| Joint | Weighted | 2.551e-03 | 6.723e-04 | 1.825e-04 |
| Joint | L2       | 2.546e-03 | 6.692e-04 | 1.809e-04 |

### Interpretation

All controlled sequences decrease for both solvers. At these resolutions,
spatial error dominates the residual, so adding time steps alone gives less
improvement than refining the log-price mesh.

## Domain expansion

### What is being tested?

This test asks whether the artificial finite boundaries materially determine
the computed price.

### Why does it matter?

A small grid error on an inadequate domain can still give the wrong
infinite-domain option price.

### How was it tested?

The log-price half-width was expanded from 2 to 3 to 4 while holding
`dx = 0.025` and `Nt = 240` fixed. Every case uses `r = 0.03`. The table
reports the absolute price change from half-width 3 to 4.

### Results

| Case                    | `alpha` | `T`  | `sigma` | Weighted change | L2 change |
| :---------------------- | ------: | ---: | ------: | --------------: | --------: |
| ATM call, short         | 0.50    | 0.10 | 0.25    | 6.9e-18         | 6.9e-18   |
| ITM call, long/high vol | 0.90    | 2.00 | 0.60    | 2.0e-08         | 1.5e-08   |
| OTM put, long           | 0.50    | 1.50 | 0.30    | 3.0e-11         | 2.0e-11   |
| ITM put, short/high vol | 0.90    | 0.25 | 0.60    | 0.0e+00         | 2.8e-17   |

### Interpretation

All width-3 to width-4 changes are below `1.0e-06`. For these tested cases, the
reported finite-difference trends are not artifacts of a visibly narrow
domain.

## `refine_price`

### What is being tested?

This test checks whether the practical refinement helper recognizes regular
convergence and refuses misleading estimates.

### Why does it matter?

Production use needs a numerical uncertainty signal, while nearly equal or
irregular prices must not be mistaken for proven accuracy.

### How was it tested?

Both solvers started from `Nx = Nt = 40` on `[-4, 4]`, with tolerance
`5.0e-04` and three allowed refinements. Separate tests used an unattainable
`1.0e-12` tolerance and an artificial irregular price sequence.

### Results

| Check                         | Result                  |
| :---------------------------- | :---------------------- |
| Weighted regular sequence     | Converged               |
| L2 regular sequence           | Converged               |
| Refinements used              | 3                       |
| Estimated remaining error     | At most `5.0e-04`       |
| Tolerance `1.0e-12`           | Correctly not converged |
| Artificial irregular sequence | Correctly rejected      |
| Finite-domain error included  | No                      |

### Interpretation

The helper behaves as an empirical discretization check: it accepts both
regular solver sequences and rejects cases where its assumptions are not met.
Domain expansion remains a separate validation responsibility.

# Remaining uncertainties

The numerical validation does not resolve every issue in the source papers.

An24 Theorem 2, PDF pages 11--13, claims unconditional stability. Between
equations (31) and (32), its proof drops a drift cross term that does not cancel
for consecutive time levels. A direct check at
$(x_L,x_R)=(0,\pi)$, `alpha = 0.50`, `dt = 0.01`, and
$\mu=r=0.01$, using $u^0=\sin(30x)$, makes the normalized left side of
equation (32) equal `5.130259...`, above the claimed bound of `4`. With
nonzero drift and two eigenmodes, the omitted cross term is `-6.7195e-05` at
step four. This invalidates the printed proof, not the numerical method; the
implementation therefore relies on direct convergence evidence.

KMP20 Example 2 is only partially reproduced. The ordering by `theta` agrees,
but on the stated `(n, N) = (500, 50)` grid, the computed errors are `0.603%`,
`0.325%`, and `0.047%` for `theta = 0`, `0.25`, and `0.50`, compared with the
paper's `1.74%`, `1.12%`, and `0.61%`. The paper does not specify how `S0 = 1`
is extracted when `log(S0) = 0` is not a grid node. Linear interpolation gives
the errors above. Using the two adjacent nodes instead
gives `2.67%` and `5.19%` for `theta = 0.50`; reversing the stated space/time
counts gives about `1.7%` for all three weights. None of these interpretations
fully reconciles the table. KMP20 Table 2 is not used as a spatial benchmark
because it measures error at the prescribed boundary `x = x_max`.

The runtime cost of the two methods has not yet been benchmarked. Both retain
the full solution history, so calibration-scale performance remains a
separate concern.

# What can we conclude right now?

The code contains two independent European TFBS finite-difference solvers.
They reproduce important source-paper results, recover analytic BSM at
`alpha = 1`, approach one another under refinement, and converge toward an
independent subordination benchmark. The checks cover a broad `alpha` range,
7- and 14-day maturities, separate space/time refinement, finite-domain
sensitivity, and behavior close to `alpha = 1`.

These results validate the **numerical pricing engine**. They do not yet show
that TFBS fits or predicts real market option prices better than classical BSM.
Calibration accuracy, out-of-sample performance, and runtime at empirical
scale are separate future questions.

# Paper-to-code citation crosswalk

| Implemented item             | Primary-source locator                                        | Code                          |
| :--------------------------- | :------------------------------------------------------------ | :---------------------------  |
| Model and `x = log(S)`       | KMP20 eqs. (4)--(6), pp. 5--6; An24 eqs. (2)--(4), pp. 4--5   | `model.py`                    |
| Weighted L1 history          | KMP20 eq. (7), p. 6; recurrence (11), p. 7                    | `weighted_solver.py`          |
| Weighted stability and order | KMP20 Theorems 3.2 and 3.3, pp. 10 and 15--16                 | `weighted_solver.py`          |
| L2 startup and coefficients  | An24 eqs. (7)--(9), p. 6                                      | `l2_solver.py`                |
| L2 space and recurrence      | An24 eqs. (12)--(17), p. 7                                    | `l2_solver.py`                |
| An24 matrix-sign discrepancy | An24 eqs. (14)--(17), p. 7, versus eq. (20), p. 8             | `l2_solver.py`                |
| Subordination benchmark      | KMP20 Section 2.2, pp. 3--5; Kanter75 stable representation   | `validation/subordination.py` |

The source locator is also repeated beside each nontrivial recurrence in the
implementation so the code can be audited directly.

# References

- **KMP20:** G. Krzyżanowski, M. Magdziarz, and Ł. Płociniczak, “A weighted
  finite difference method for subdiffusive Black-Scholes model,” *Computers &
  Mathematics with Applications* **80**(5), 653--670 (2020).
  [Publisher DOI](https://doi.org/10.1016/j.camwa.2020.04.029);
  [open manuscript, arXiv:1907.00297v4](https://arxiv.org/abs/1907.00297v4).

- **An24:** X. An, Q. Wang, F. Liu, V. V. Anh, and I. W. Turner, “Parameter
  estimation for time-fractional Black-Scholes equation with S&P 500 index
  option,” *Numerical Algorithms* **95**, 1--30 (2024). Published online
  27 June 2023.
  [Publisher DOI and open article](https://doi.org/10.1007/s11075-023-01563-4).

- **Kanter75:** M. Kanter, “Stable densities under change of scale and total
  variation inequalities,” *The Annals of Probability* **3**(4), 697--707
  (1975). [Publisher DOI](https://doi.org/10.1214/aop/1176996309).
