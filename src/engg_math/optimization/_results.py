"""Typed result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Array


@dataclass(frozen=True, slots=True)
class ScalarMinimum:
    """Minimum of a function of one variable."""

    x: float
    fun: float
    iterations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class Minimum:
    """Minimum of a scalar field; ``grad_norm`` is the 2-norm of the gradient at ``x``."""

    x: Array
    fun: float
    grad_norm: float
    iterations: int
    evaluations: int
    converged: bool
    method: str


@dataclass(frozen=True, slots=True)
class LeastSquaresFit:
    """Nonlinear least-squares solution minimising ``0.5 * ||r(x)||^2``.

    ``covariance`` is ``(J^T J)^-1`` from the SVD of the Jacobian at ``x`` (parameter
    covariance for unit residual variance); ``None`` if ``J`` is rank deficient.
    """

    x: Array
    cost: float
    residual_norm: float
    evaluations: int
    converged: bool
    covariance: Array | None


@dataclass(frozen=True, slots=True)
class LinearProgramSolution:
    """Solution of a linear program."""

    x: Array
    fun: float
    iterations: int
