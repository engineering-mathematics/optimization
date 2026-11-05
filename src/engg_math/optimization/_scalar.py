"""Minimisation of a function of one variable on a bracketing interval."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy import optimize

from ._errors import ConvergenceError, DomainError
from ._results import ScalarMinimum

if TYPE_CHECKING:
    from ._types import ScalarFunction

DEFAULT_XTOL = 1e-10
DEFAULT_MAX_ITER = 500


def minimize_scalar(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    xtol: float = DEFAULT_XTOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> ScalarMinimum:
    """Bounded Brent minimisation of ``f`` on ``[a, b]``.

    Finds a local minimum inside the interval; for a multimodal ``f`` the result depends
    on the interval. A minimum at an endpoint is reported as such, not extrapolated.
    """
    if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
        raise DomainError(f"need finite a < b, got a={a}, b={b}")
    if xtol <= 0 or max_iter < 1:
        raise DomainError(f"need xtol > 0 and max_iter >= 1; got {xtol}, {max_iter}")
    outcome = optimize.minimize_scalar(
        f, bounds=(a, b), method="bounded", options={"xatol": xtol, "maxiter": max_iter}
    )
    if not outcome.success:
        raise ConvergenceError(f"minimize_scalar: {outcome.message}")
    return ScalarMinimum(float(outcome.x), float(outcome.fun), int(outcome.nit), True)
