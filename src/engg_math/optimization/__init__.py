"""Minimisation, least squares and linear programming with typed, explicit results."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._errors import ConvergenceError, DomainError, InfeasibleError, UnboundedError
from ._least_squares import least_squares
from ._linprog import linprog
from ._multivariate import minimize
from ._results import LeastSquaresFit, LinearProgramSolution, Minimum, ScalarMinimum
from ._scalar import minimize_scalar

try:
    __version__ = version("engg-math-optimization")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "ConvergenceError",
    "DomainError",
    "InfeasibleError",
    "LeastSquaresFit",
    "LinearProgramSolution",
    "Minimum",
    "ScalarMinimum",
    "UnboundedError",
    "__version__",
    "least_squares",
    "linprog",
    "minimize",
    "minimize_scalar",
]
