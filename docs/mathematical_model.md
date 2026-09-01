# Mathematical Model

## Locked model family

The candidate model is the subdiffusive time-fractional Black-Scholes model associated with an inverse-stable market-time clock. Its fractional order satisfies:

```text
0 < alpha <= 1
```

The boundary `alpha = 1` must recover the selected classical BSM implementation under identical inputs.

This repository does not use `alpha` as a Hurst exponent and does not equate the model with fractional Brownian motion or rough volatility.

## Numerical acceptance requirements

A TFBSM pricer is not accepted merely because it produces plausible-looking option values. Before empirical use it must satisfy all of the following:

1. The governing equation and terminal/boundary conditions are traced to the locked literature definition.
2. The numerical method is documented with its discretization, convergence assumptions, and failure modes.
3. The `alpha = 1` limit agrees with the same-input BSM implementation within a declared, justified numerical tolerance.
4. Grid refinement is reported.
5. At least one independent numerical representation or published example is used as a cross-check.
6. Solver failures and numerical uncertainty remain visible in model outputs.

## Deliberately unresolved

No production pricer is implemented yet. The exact fractional operator convention, solver, boundary truncation, and validation tolerances will not be selected by convenience or copied from the unvalidated prototype. They remain research decisions requiring direct literature and numerical evidence.
