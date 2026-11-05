"""Nonlinear least squares with a parameter covariance from the Jacobian's SVD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from engg_math.core import ShapeError
from engg_math.numerical_computation import jacobian as numerical_jacobian

from ._errors import ConvergenceError, DomainError
from ._results import LeastSquaresFit

if TYPE_CHECKING:
    from ._types import Array, VectorFunction

DEFAULT_TOL = 1e-10
DEFAULT_MAX_EVALS = 1000
RANK_RTOL = 1e-12


def _covariance(jac: Array) -> Array | None:
    """``(J^T J)^-1`` via SVD; ``None`` when ``J`` lacks full column rank."""
    _, sing, vt = np.linalg.svd(jac, full_matrices=False)
    if sing.size < jac.shape[1] or sing[-1] <= RANK_RTOL * sing[0]:
        return None
    inv_sq = (1.0 / sing) ** 2
    cov: Array = (vt.T * inv_sq) @ vt
    return cov


def least_squares(
    residuals: VectorFunction,
    x0: Array | list[float],
    jacobian: VectorFunction | None = None,
    *,
    tol: float = DEFAULT_TOL,
    max_evals: int = DEFAULT_MAX_EVALS,
    raise_on_failure: bool = True,
) -> LeastSquaresFit:
    """Minimise ``0.5 * ||r(x)||^2`` (trust-region reflective).

    ``jacobian`` returns the ``m x n`` matrix of ``r``; it defaults to central
    differences. ``tol`` applies to the cost, step and gradient tests.
    """
    if tol <= 0 or max_evals < 1:
        raise DomainError(f"need tol > 0 and max_evals >= 1; got {tol}, {max_evals}")
    start = np.array(x0, dtype=float)
    if start.ndim != 1:
        raise ShapeError(f"x0 must be 1-D, got shape {start.shape}")

    def jac_at(x: Array) -> Array:
        if jacobian is not None:
            return np.asarray(jacobian(x), dtype=float)
        return np.asarray(numerical_jacobian(residuals, x).data, dtype=float)

    outcome = optimize.least_squares(
        residuals, start, jac=jac_at, method="trf", xtol=tol, ftol=tol, gtol=tol, max_nfev=max_evals
    )
    if not outcome.success and raise_on_failure:
        raise ConvergenceError(f"least_squares: {outcome.message}")
    x = np.asarray(outcome.x, dtype=float)
    return LeastSquaresFit(
        x=x,
        cost=float(outcome.cost),
        residual_norm=float(np.linalg.norm(outcome.fun)),
        evaluations=int(outcome.nfev),
        converged=bool(outcome.success),
        covariance=_covariance(np.asarray(outcome.jac, dtype=float)),
    )
