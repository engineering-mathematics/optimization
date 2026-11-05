"""Unconstrained minimisation of a scalar field."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import optimize

from engg_math.core import ShapeError
from engg_math.numerical_computation import gradient as numerical_gradient

from ._errors import ConvergenceError, DomainError
from ._results import Minimum

if TYPE_CHECKING:
    from ._types import Array, GradientFunction, ScalarField

Method = Literal["bfgs", "lbfgs", "nelder-mead"]
_SCIPY_NAME = {"bfgs": "BFGS", "lbfgs": "L-BFGS-B", "nelder-mead": "Nelder-Mead"}
DEFAULT_TOL = 1e-8
DEFAULT_MAX_ITER = 1000


def minimize(
    f: ScalarField,
    x0: Array | list[float],
    grad: GradientFunction | None = None,
    *,
    method: Method = "bfgs",
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    raise_on_failure: bool = True,
) -> Minimum:
    """Minimise ``f`` from ``x0``.

    ``bfgs`` and ``lbfgs`` stop when the gradient 2-norm is at most ``tol``; they use
    ``grad`` or, if omitted, a central-difference gradient. ``nelder-mead`` is derivative
    free and stops when simplex size and function spread are below ``tol``.
    Non-convergence raises ``ConvergenceError`` unless ``raise_on_failure`` is false.
    """
    if method not in _SCIPY_NAME:
        raise DomainError(f"method must be one of {sorted(_SCIPY_NAME)}, got {method!r}")
    if tol <= 0 or max_iter < 1:
        raise DomainError(f"need tol > 0 and max_iter >= 1; got {tol}, {max_iter}")
    start = np.array(x0, dtype=float)
    if start.ndim != 1:
        raise ShapeError(f"x0 must be 1-D, got shape {start.shape}")

    def gradient_at(x: Array) -> Array:
        return np.asarray(grad(x), dtype=float) if grad else numerical_gradient(f, x)

    if method == "nelder-mead":
        options = {"xatol": tol, "fatol": tol, "maxiter": max_iter}
    else:
        options = {"gtol": tol, "maxiter": max_iter}
    # scipy-stubs overloads do not model a method chosen at runtime.
    outcome = optimize.minimize(  # type: ignore[call-overload]
        f,
        start,
        jac=None if method == "nelder-mead" else gradient_at,
        method=_SCIPY_NAME[method],
        options=options,
    )
    x = np.asarray(outcome.x, dtype=float)
    result = Minimum(
        x=x,
        fun=float(outcome.fun),
        grad_norm=float(np.linalg.norm(gradient_at(x))),
        iterations=int(outcome.nit),
        evaluations=int(outcome.nfev),
        converged=bool(outcome.success),
        method=method,
    )
    if not result.converged and raise_on_failure:
        raise ConvergenceError(f"{method}: {outcome.message} (||grad||={result.grad_norm:.3e})")
    return result
