# Numerical solver definitions and paper crosswalk

This project implements two independent finite-difference solvers for the same
European time-fractional Black--Scholes problem.  This note fixes the model and
notation before code is used, and records where each numerical formula comes
from.

## Canonical pricing problem

Let `t` be elapsed time from valuation toward maturity, measured in years, and
let `x = log(S)`.  For `0 < alpha <= 1`, both papers reduce to

```text
Caputo_D_t^alpha u = (sigma^2 / 2) u_xx
                     + (r - sigma^2 / 2) u_x - r u,
u(x, 0) = payoff(exp(x)).
```

The Caputo derivative is

```text
Caputo_D_t^alpha g(t)
  = 1 / Gamma(1-alpha) * integral_0^t g'(s) / (t-s)^alpha ds.
```

This is Krzyzanowski, Magdziarz, and Plociniczak (2020), equations (4)--(6),
pages 5--6, and An et al. (2024), equations (2)--(4), pages 4--5.  The
stochastic model in the former is a geometric Brownian motion evaluated at the
inverse of an independent alpha-stable subordinator.  Both papers assume
constant `r` and `sigma` and zero dividend yield.  The implementation therefore
does not add a dividend term.

Krzyzanowski et al. define the inverse clock in equation (1), page 2, assume it
is independent of the Brownian motion, and select a minimal-relative-entropy
martingale measure in Proposition 2.1, page 3. An et al. do not state an
underlying stochastic clock or a martingale measure; they introduce their model
on page 4 by replacing the ordinary time derivative in BSM with a Caputo
derivative. The canonical interface therefore standardizes only the PDE the two
papers actually share, not an unstated stochastic interpretation for An et al.

An et al. include a dimensional coefficient `rho` with units
`year^(alpha-1)` and then set `rho = 1` (equation (4), page 5).  The canonical
interface uses years and makes the same normalization.  This convention is
material: changing the time unit without rescaling `rho` changes the model.
The fractional order `alpha` itself is dimensionless.

At `alpha = 1`, the Caputo derivative becomes the ordinary first derivative,
so the canonical equation is the classical Black--Scholes equation in forward
time-to-maturity coordinates.

## Initial and boundary data

On a finite log-price interval `[x_min, x_max]`, the code uses the standard
European asymptotic boundaries at elapsed time `t`:

```text
call: u(x_min,t) = 0
      u(x_max,t) = exp(x_max) - K exp(-r t)

put:  u(x_min,t) = K exp(-r t) - exp(x_min)
      u(x_max,t) = 0
```

The call boundaries agree with Krzyzanowski et al., page 6, after translating
their elapsed-time notation.  An et al. equations (5)--(6), page 5, print
`x_R` where dimensional consistency requires `exp(x_R)`, and use `T-t` in a
place where their transformation `u(x,t)=V(S,T-t)` requires elapsed time `t`.
Those expressions are treated as typographical errors rather than implemented
literally.  The finite-left put boundary retains the small `exp(x_min)` term;
the papers state the limiting boundary as `x_min -> -infinity`.

There is also a model-level inconsistency worth preserving in the record.  The
put--call parity in Krzyzanowski et al., Proposition 2.1, page 3, and the
exponential strike discount used in both papers' boundary data imply
`C-P = S-K exp(-rt)`.  For `alpha < 1`, that function does not satisfy the
printed Caputo PDE because the Caputo derivative of `exp(-rt)` is not its
ordinary derivative.  The homogeneous PDE instead gives a Mittag--Leffler
discount factor.  The implementation keeps the papers' stated exponential
boundaries so both numerical methods solve the same published finite-domain
problem, and the validation report exposes the resulting fractional-parity
residual.  At `alpha=1` the inconsistency disappears.

## Solver A: weighted L1 finite differences

`tfbsm_pricing.weighted_fd` follows Krzyzanowski et al. equations (7)--(11),
pages 6--7.  With `dt = T/N`,

```text
b_j = (j+1)^(1-alpha) - j^(1-alpha),
d   = Gamma(2-alpha) dt^alpha.
```

Centered differences discretize the spatial operator.  In the paper's
weighting convention, `theta = 0` is fully implicit and `theta = 1` is fully
explicit.  The default is their optimal stable weight

```text
theta_hat = (2 - 2^(1-alpha)) / (3 - 2^(1-alpha)),
```

from page 16.  At `alpha = 1`, it equals `1/2` and the method reduces exactly
to Crank--Nicolson.  The paper claims global error
`O(dt^(2-alpha) + dx^2)` in Theorem 3.3, page 15.  Its stability result is
unconditional only in the parameter region stated in Theorem 3.2, pages
10--12; the optimal default lies in that region.

Writing `L_h` for the centered spatial operator and `G^k` for its boundary
contribution, the implemented equation (11) is

```text
[I-(1-theta)d L_h] U^n
  = sum_(j=0)^(n-2) (b_j-b_(j+1)) U^(n-1-j) + b_(n-1) U^0
    + theta d L_h U^(n-1)
    + (1-theta) G^n + theta G^(n-1).
```

For `n=1`, the history sum is simply `U^0`. This form also records the precise
newest-to-oldest indexing used by the code.

## Solver B: L2 finite differences

`tfbsm_pricing.l2_fd` independently implements An et al. equations (7)--(17),
pages 6--7.  The first time step uses their L1 formula with

```text
phi_1 = Gamma(2-alpha) dt^alpha.
```

Later steps use the paper's quadratic L2 interpolation with the three
coefficient sequences `a_i`, `b_i`, and `c_i`, and

```text
phi_2 = Gamma(3-alpha) dt^alpha.
```

The sequences from equations (8)--(9), page 6, are

```text
a_i = (2-alpha)[i^(1-alpha)/2 - 3(i+1)^(1-alpha)/2]
      - i^(2-alpha) + (i+1)^(2-alpha),
b_i = (4-2alpha)(i+1)^(1-alpha)
      + 2i^(2-alpha) - 2(i+1)^(2-alpha),
c_i = (alpha/2-1)[i^(1-alpha)+(i+1)^(1-alpha)]
      - i^(2-alpha) + (i+1)^(2-alpha).
```

The complete time-history recurrence is implemented directly from equations
(14)--(17), page 7. Centered first and second spatial differences are equations
(12)--(13), page 7.

These weights are implemented in the L2 module rather than shared with the L1
solver, so a mistake in one time discretization cannot automatically reproduce
itself in the other.  At `alpha = 1`, the later steps reduce exactly to BDF2,
with a backward-Euler startup.

The component equations (14)--(17) determine the implemented tridiagonal
signs.  The matrix displayed in equation (20), page 8, has a positive
right-off-diagonal entry that conflicts with those component equations and
with the stated differential operator; the code does not copy that sign.

An et al. claim temporal order `3-alpha` and spatial order two.  Their first
step is only L1, and equation (39), page 14, says it can be transformed to the
higher-order form without giving the transformation.  The implementation
faithfully uses the published equation (14) startup and does not invent the
missing correction.  Validation therefore reports the measured order of the
published recurrence rather than presuming the claimed order.

## Stability-proof limitation in An et al.

Theorem 2 in An et al., pages 11--13, claims unconditional stability in the
ordinary discrete `L2` norm.  In the move from equation (31) to equation (32),
the drift cross-term is discarded even though it does not cancel for the
forward-difference pairing used in their proof.  A bounded grid function gives
a direct counterexample to the asserted intermediate inequality.  This does
not by itself show that the numerical recurrence is unstable; it means the
published proof does not establish the stated theorem.  Solver validation
therefore relies on convergence, benchmark, and cross-solver evidence rather
than treating that proof as conclusive.

A concrete check of equation (32) uses `(x_L,x_R)=(0,pi)`, `alpha=1/2`,
`dt=1/100`, `mu=r=1/100`, and `u^0=sin(30x)`. The exact first semidiscrete step
makes the left side of the claimed bound, divided by `||u^0||_2`, equal to
`5.130259...`, while equation (32) bounds it by `4`. Separately, with nonzero
drift and a two-eigenmode initial condition, the cross term discarded in the
paper evaluates to `-6.7195e-5` at step four rather than zero. These checks
invalidate the proof steps, not the numerical method itself.

## Independence and comparison rules

The solvers share only the model, grid, boundary definitions, result container,
and tridiagonal linear algebra.  Their Caputo-history weights, startup rules,
and time recurrences are separate.  Cross-solver comparisons always use the
same model, domain, spatial grid, time grid, payoff, and corrected boundaries.
Agreement between solvers is supporting numerical evidence, not an analytic
proof of correctness.

## Indexing and sign audit

The code's zero-based array level `surface[n]` represents the papers' `U^n` or
`u^n`. In the weighted recurrence, `b_j-b_(j+1)` multiplies
`surface[n-1-j]`, and `b_(n-1)` multiplies the payoff `surface[0]`, exactly as
in Krzyzanowski et al. equation (11), page 7. In the L2 recurrence, the lag
`n-j` in An et al. equation (17), page 7, is kept distinct from the stored time
level `j`; the three special startup equations are coded separately rather
than folded into a general loop.

For `L = a*d_xx + b*d_x - r`, the centered spatial row is

```text
lower = a/dx^2 - b/(2dx)
diag  = -2a/dx^2 - r
upper = a/dx^2 + b/(2dx).
```

Consequently the weighted left side is `I-(1-theta)dL`, the first L2 left side
is `I-phi_1 L`, and later L2 left sides are `beta I-phi_2 L`. Boundary entries
removed from those matrices enter the right side with a positive multiplier.
This agrees with Krzyzanowski et al. equations (7)--(11), pages 6--7, and the
component equations (14)--(17) in An et al., page 7. It also explains why the
conflicting upper-diagonal sign in An et al. equation (20), page 8, is rejected.

The weighting convention was checked independently: Krzyzanowski et al.
`theta=0` is implicit and `theta=1` is explicit. The code never substitutes the
more common opposite theta convention.

## Reproducibility targets

Validation covers:

1. the European-call numerical examples in Krzyzanowski et al., pages 17--18;
2. the manufactured-solution table in An et al., pages 13--14;
3. exact `alpha = 1` comparisons with the analytic Black--Scholes price;
4. grid refinement in time and space;
5. cross-solver agreement for several fractional orders and both calls and
   puts;
6. payoff, boundary, non-negativity, monotonicity, and put--call parity checks.

The validation report records any paper result that cannot be reproduced from
the published information, including the unspecified first-step
transformation in An et al.
