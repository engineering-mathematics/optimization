"""Reference-solution, property and error-path tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engg_math.core import ShapeError
from engg_math.optimization import (
    ConvergenceError,
    DomainError,
    InfeasibleError,
    UnboundedError,
    least_squares,
    linprog,
    minimize,
    minimize_scalar,
)


def rosenbrock(v: np.ndarray) -> float:
    return float((1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2)


def rosenbrock_grad(v: np.ndarray) -> np.ndarray:
    return np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2), 200 * (v[1] - v[0] ** 2)])


# --- scalar ----------------------------------------------------------------------------
def test_minimize_scalar_parabola() -> None:
    result = minimize_scalar(lambda x: (x - 1.5) ** 2 + 2, -5.0, 5.0)
    assert result.x == pytest.approx(1.5, abs=1e-6)  # Brent limited to ~sqrt(eps) in x
    assert result.fun == pytest.approx(2.0, abs=1e-10)


def test_minimize_scalar_rejects_bad_input() -> None:
    with pytest.raises(DomainError):
        minimize_scalar(math.sin, 1.0, 0.0)
    with pytest.raises(DomainError):
        minimize_scalar(math.sin, 0.0, math.inf)
    with pytest.raises(DomainError):
        minimize_scalar(math.sin, 0.0, 1.0, xtol=0.0)


def test_minimize_scalar_raises_when_iterations_exhausted() -> None:
    with pytest.raises(ConvergenceError):
        minimize_scalar(lambda x: (x - 0.3) ** 2, -5.0, 5.0, xtol=1e-14, max_iter=2)


# --- multivariate ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["bfgs", "lbfgs"])
def test_gradient_methods_solve_rosenbrock_with_analytic_gradient(method: str) -> None:
    result = minimize(rosenbrock, [-1.2, 1.0], rosenbrock_grad, method=method)  # type: ignore[arg-type]
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)
    assert result.converged
    assert result.grad_norm < 1e-3  # L-BFGS-B's gtol bounds the projected gradient only


def test_bfgs_with_numerical_gradient() -> None:
    result = minimize(lambda v: float(np.sum((v - [1.0, -2.0]) ** 2)), [0.0, 0.0])
    np.testing.assert_allclose(result.x, [1.0, -2.0], atol=1e-6)


def test_nelder_mead_derivative_free() -> None:
    result = minimize(
        lambda v: float(np.sum((v - 3.0) ** 2)), [0.0, 0.0], method="nelder-mead", tol=1e-9
    )
    np.testing.assert_allclose(result.x, [3.0, 3.0], atol=1e-4)


def test_minimize_failure_modes() -> None:
    with pytest.raises(ConvergenceError):
        minimize(rosenbrock, [-1.2, 1.0], max_iter=2)
    result = minimize(rosenbrock, [-1.2, 1.0], max_iter=2, raise_on_failure=False)
    assert not result.converged


def test_minimize_validates_input() -> None:
    with pytest.raises(DomainError):
        minimize(rosenbrock, [0.0, 0.0], method="newton")  # type: ignore[arg-type]
    with pytest.raises(DomainError):
        minimize(rosenbrock, [0.0, 0.0], tol=0.0)
    with pytest.raises(ShapeError):
        minimize(rosenbrock, np.ones((2, 2)))


@given(st.lists(st.floats(-10, 10), min_size=1, max_size=4))
@settings(max_examples=30, deadline=None)
def test_bfgs_finds_minimum_of_shifted_quadratic(centre: list[float]) -> None:
    """REQ-OPT-001: the minimiser of sum((x - c)^2) is c."""
    target = np.array(centre)
    result = minimize(lambda v: float(np.sum((v - target) ** 2)), np.zeros_like(target))
    np.testing.assert_allclose(result.x, target, atol=1e-5)


# --- least squares ---------------------------------------------------------------------
def test_least_squares_exponential_fit_recovers_parameters() -> None:
    t = np.linspace(0, 3, 40)
    truth = np.array([2.5, 0.8])
    data = truth[0] * np.exp(-truth[1] * t)

    fit = least_squares(lambda p: p[0] * np.exp(-p[1] * t) - data, [1.0, 0.1])
    np.testing.assert_allclose(fit.x, truth, atol=1e-7)
    assert fit.residual_norm < 1e-7
    assert fit.covariance is not None
    assert fit.covariance.shape == (2, 2)


def test_least_squares_analytic_jacobian_matches() -> None:
    t = np.linspace(0, 1, 10)
    data = 3.0 * t + 1.0

    fit = least_squares(
        lambda p: p[0] * t + p[1] - data,
        [0.0, 0.0],
        jacobian=lambda p: np.column_stack([t, np.ones_like(t)]),
    )
    np.testing.assert_allclose(fit.x, [3.0, 1.0], atol=1e-8)


def test_least_squares_rank_deficient_has_no_covariance() -> None:
    t = np.linspace(0, 1, 10)
    fit = least_squares(lambda p: (p[0] + p[1]) * t - t, [0.0, 0.0])
    assert fit.covariance is None


def test_least_squares_validates_and_reports_failure() -> None:
    with pytest.raises(DomainError):
        least_squares(lambda p: p, [0.0], tol=0.0)
    with pytest.raises(ShapeError):
        least_squares(lambda p: p, np.ones((1, 2)))
    with pytest.raises(ConvergenceError):
        least_squares(lambda p: np.array([p[0] ** 4 - 1, p[1] ** 3]), [5.0, 5.0], max_evals=1)


# --- linear programming ----------------------------------------------------------------
def test_linprog_reference_solution() -> None:
    """min -x - 2y s.t. x + y <= 4, x <= 3, x, y >= 0 -> (0, 4), objective -8."""
    sol = linprog([-1, -2], A_ub=[[1, 1], [1, 0]], b_ub=[4, 3])
    np.testing.assert_allclose(sol.x, [0.0, 4.0], atol=1e-9)
    assert sol.fun == pytest.approx(-8.0)


def test_linprog_equality_constraint() -> None:
    sol = linprog([1, 1], A_eq=[[1, 2]], b_eq=[4])
    assert sol.fun == pytest.approx(2.0)


def test_linprog_infeasible_and_unbounded() -> None:
    with pytest.raises(InfeasibleError):
        linprog([1.0], A_ub=[[1.0]], b_ub=[-1.0])
    with pytest.raises(UnboundedError):
        linprog([-1.0])


def test_linprog_validates_input() -> None:
    with pytest.raises(ShapeError):
        linprog([[1.0]])
    with pytest.raises(ShapeError):
        linprog([1.0, 2.0], A_ub=[[1.0]], b_ub=[1.0])
    with pytest.raises(DomainError):
        linprog([1.0], A_ub=[[1.0]])
