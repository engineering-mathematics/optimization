"""Optimisation errors."""

from __future__ import annotations

from engg_math.core import EnggMathError, NumericalError


class ConvergenceError(NumericalError):
    """The optimiser stopped without meeting its tolerance."""


class InfeasibleError(NumericalError):
    """The constraints admit no solution."""


class UnboundedError(NumericalError):
    """The objective decreases without limit over the feasible set."""


class DomainError(EnggMathError, ValueError):
    """An argument is outside the routine's valid domain."""
