"""Linear programming (HiGHS) with explicit infeasible/unbounded errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from engg_math.core import ShapeError

from ._errors import ConvergenceError, DomainError, InfeasibleError, UnboundedError
from ._results import LinearProgramSolution

if TYPE_CHECKING:
    import numpy.typing as npt

STATUS_ITERATION_LIMIT = 1
STATUS_INFEASIBLE = 2
STATUS_UNBOUNDED = 3


def _matrix(name: str, value: npt.ArrayLike | None, columns: int) -> npt.NDArray[np.float64] | None:
    if value is None:
        return None
    arr = np.atleast_2d(np.asarray(value, dtype=float))
    if arr.ndim != 2 or arr.shape[1] != columns:  # noqa: PLR2004
        raise ShapeError(f"{name} must have {columns} columns, got shape {arr.shape}")
    return arr


def linprog(
    c: npt.ArrayLike,
    A_ub: npt.ArrayLike | None = None,  # noqa: N803
    b_ub: npt.ArrayLike | None = None,
    A_eq: npt.ArrayLike | None = None,  # noqa: N803
    b_eq: npt.ArrayLike | None = None,
    *,
    bounds: tuple[float | None, float | None] = (0.0, None),
) -> LinearProgramSolution:
    """Minimise ``c @ x`` subject to ``A_ub x <= b_ub``, ``A_eq x == b_eq`` and bounds.

    ``bounds`` applies to every variable and defaults to ``x >= 0``. Raises
    ``InfeasibleError`` or ``UnboundedError`` instead of returning a flagged result.
    """
    cost = np.asarray(c, dtype=float)
    if cost.ndim != 1 or cost.size == 0:
        raise ShapeError(f"c must be a non-empty 1-D vector, got shape {cost.shape}")
    ub, eq = _matrix("A_ub", A_ub, cost.size), _matrix("A_eq", A_eq, cost.size)
    if (ub is None) != (b_ub is None) or (eq is None) != (b_eq is None):
        raise DomainError("each constraint matrix needs its right-hand side, and vice versa")
    outcome = optimize.linprog(  # type: ignore[call-overload, misc]  # stubs reject optional matrices
        cost, A_ub=ub, b_ub=b_ub, A_eq=eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if outcome.status == STATUS_INFEASIBLE:
        raise InfeasibleError(f"linprog: {outcome.message}")
    if outcome.status == STATUS_UNBOUNDED:
        raise UnboundedError(f"linprog: {outcome.message}")
    if not outcome.success:
        raise ConvergenceError(f"linprog: {outcome.message}")
    return LinearProgramSolution(
        np.asarray(outcome.x, dtype=float), float(outcome.fun), int(outcome.nit)
    )
