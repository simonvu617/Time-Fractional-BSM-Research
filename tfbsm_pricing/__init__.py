"""Independent finite-difference solvers for European TFBS options."""

from .bsm import black_scholes_price
from .diagnostics import SolverResult
from .grid import GridSpec
from .l2_fd import solve_l2
from .model import EuropeanOptionProblem
from .weighted_fd import optimal_weight, solve_weighted

__all__ = [
    "EuropeanOptionProblem",
    "GridSpec",
    "SolverResult",
    "black_scholes_price",
    "optimal_weight",
    "solve_l2",
    "solve_weighted",
]
