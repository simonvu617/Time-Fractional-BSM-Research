# Third solver: Zhou--Gao American put

`solve_zhou_gao` adds American puts with continuous dividend yield `q`.
The existing weighted L1 and L2 solvers continue to price zero-dividend
European calls and puts. This third solver shares L1 time discretization
with the fully implicit weighted method, but uses stock space and solves
for early exercise; it is not a third independent time approximation.

The source is Zhiqiang Zhou and Xuemei Gao, **Numerical Methods for Pricing American
Options with Time-Fractional PDE Models**, *Mathematical Problems in
Engineering*, vol. 2016, Article 5614950, pp. 1--8 (2016):
[DOI](https://doi.org/10.1155/2016/5614950),
[open PDF](https://pdfs.semanticscholar.org/294d/bb2b5113e9346fd8e10b7fbeab615ffa75b3.pdf).
The implementation follows Section 4's finite-difference method, with the
adaptations below.

The paper is also listed under `references` in the repository's
`CITATION.cff`; the software's authorship metadata remains separate from
the paper's authorship. See the README for the AI authorship disclosure.

## Usage

```python
from tfbsm_pricing import AmericanPutProblem, StockGridSpec, solve_zhou_gao

problem = AmericanPutProblem(
    S0=100, K=100, T=1, r=0.05, sigma=0.20, alpha=0.80, q=0.02
)
grid = StockGridSpec(S_max=400, space_steps=400, time_steps=200)
result = solve_zhou_gao(problem, grid)

print(result.price)
print(result.exercise_boundary[-1])
```

`stock_grid` contains stock prices, including zero. `time_grid` contains
time to expiry, from zero to `T`; `grid_price[n, j]` is the value at those
two coordinates. The scalar price interpolates the final row at `S0`.
`exercise_boundary[n]` is the last exercise node at positive time, with
resolution `grid.dS`; the expiry entry is `K` by convention. When `q > r`,
the positive-time boundary need not approach `K` as expiry approaches.

Inputs require `0 < alpha <= 1`, positive `K`, `T`, and `sigma`, and
nonnegative `S0`, `r`, and `q`. Choose `S_max > K` and `S0 <= S_max`.
All step counts are numbers of intervals. Time is in years, with the same
normalized fractional-time scale as the European solvers.

The separate `AmericanPutProblem`, `StockGridSpec`, and `AmericanPutResult`
types describe the exercise contract and stock coordinates explicitly.
The existing European `GridSpec` and `refine_price` helper do not accept
this solver. To refine it, increase `space_steps` and `time_steps` on the
same stock domain; check domain size separately by increasing `S_max` while
keeping `dS` approximately fixed.

## PDE and exercise conditions

Write $\tau=T-t$ for time to expiry and $g(S)=\max(K-S,0)$.
The paper's $\beta=0$, $\delta=\sigma$ specialization has operator

```math
\mathcal L_q V=
\frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}
+(r-q)S\frac{\partial V}{\partial S}-rV.
```

The American obstacle conditions are

```math
V\ge g,\qquad
R={}^C D_\tau^\alpha V-\mathcal L_q V\ge0,\qquad
(V-g)R=0.
```

In the continuation region, $V>g$ and the PDE holds with equality.
In the exercise region, $V=g$. The initial and outer boundaries are

```math
V(S,0)=g(S),\qquad V(0,\tau)=K,\qquad V(S_{\max},\tau)=0.
```

The last condition approximates the infinite-stock boundary. The moving
boundary has value matching $V(S_f,\tau)=K-S_f$ and smooth fit
$V_S(S_f,\tau)=-1$. The algorithm approximates this boundary on the mesh.
Nonnegative interest rates are part of the supported contract because
immediate exercise gives the left boundary $K$.

This implements the specified Caputo obstacle PDE. Interpreting `q` in the
inverse-stable framework means a continuous yield on the same operational
clock as the stock and discounting. Scheduled cash dividends and stopping
problems with additional clock-age state are outside this interface. The
numerical checks do not prove equivalence to every inverse-clock stopping
model.

## Discretization and source mapping

The stock mesh is uniform on $[0,S_{\max}]$, with full time history retained
at every node, including values from the exercise region. At time step $n$,
define

```math
d=\Gamma(2-\alpha)\Delta\tau^\alpha,\qquad
w_k=(k+1)^{1-\alpha}-k^{1-\alpha},
```

```math
H_j^n=V_j^{n-1}-\sum_{k=1}^{n-1}w_k
\left(V_j^{n-k}-V_j^{n-k-1}\right).
```

At `alpha=1`, $w_0=1$ and $w_k=0$ for $k>0$, giving backward Euler.
For the central stock stencil, put

```math
a_j=\frac{\sigma^2 S_j^2}{2\Delta S^2},\qquad
b_j=\frac{(r-q)S_j}{2\Delta S}.
```

The continuation solve is

```math
-d(a_j-b_j)V_{j-1}^n
+[1+d(2a_j+r)]V_j^n
-d(a_j+b_j)V_{j+1}^n=H_j^n.
```

| Item | Source | Implementation |
| --- | --- | --- |
| Put PDE and free boundary | Equations (10)--(14), p. 3 | `AmericanPutProblem`, exercise contract |
| Uniform stock and time grids | Equation (34), p. 5 | `StockGridSpec`, grid construction |
| L1 memory and central stock differences | Equations (35)--(39), p. 5 | `solve_zhou_gao` |
| Moving exercise boundary | Step 1 / (40), Step 2.1, p. 5 | Backward search with continuation solves |

Three implementation details matter:

1. The printed diagonal in (39) omits $r\Delta\tau$. The code restores it
   to match the $-rV$ term of PDE (10); after normalization this is the
   $dr$ contribution above.
2. The boundary search retains the adjacent feasible continuation solution
   when a trial falls below payoff. It uses the mesh boundary; the paper's
   optional quadratic subgrid refinement, Step 2.2 / (41)--(44), is not
   implemented. A European solve followed by clipping would not satisfy
   the same coupled exercise problem.
3. If a central coefficient $a_j-b_j$ or $a_j+b_j$ is negative, that row
   uses a one-sided drift derivative with nonnegative spatial off-diagonal
   coefficients. This can occur near zero stock even on refined meshes.
   `one_sided_stock_nodes` records the affected nodes. These rows have
   first-order drift accuracy; the central stencil is unchanged elsewhere.

The source is linked beside the corresponding formulas in
`tfbsm_pricing/zhou_gao_solver.py`. No Laplace-transform approximation,
general CEV elasticity, American call, or discrete cash-dividend handling
is included.

## Checks and limits of the evidence

`tests/test_zhou_gao_solver.py` checks the ordinary American limit against
an independent 2,000-step CRR tree, with and without dividends. For
$S_0=K=100$, $T=1$, $r=0.05$, $\sigma=0.2$, and $\alpha=1$:

| Dividend yield | CRR tree | Grid 200 x 100 | Grid 400 x 200 |
| --- | ---: | ---: | ---: |
| 0.00 | 6.089990 | 6.065722 | 6.080604 |
| 0.08 | 8.954212 | 8.933650 | 8.946738 |

Both grids use `S_max=400`; counts are stock intervals x time intervals.
These are measured examples, not a general one-cent error guarantee.

The fractional tests reconstruct the Caputo derivative and spatial
operator directly to check payoff, boundaries, and complementarity over
the full surface. A zero-rate, zero-dividend case, where early exercise
has no benefit, agrees with the independent inverse-stable European
benchmark. Additional cases check the early-exercise premium and dividend
response, including `q > r` and a strike between mesh nodes.

Diagnostics report obstacle and scaled PDE residuals, using
$A V^n-H^n=dR^n$. Small residuals show that the discrete problem was solved;
they do not measure continuum discretization or far-boundary error.
There is no general fractional American pricing error bound established
by these tests, and no claim of reproducing the paper's numerical tables.
