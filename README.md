# engg-math-optimization

Minimisation, nonlinear least squares and linear programming for `engg_math`. Thin, typed layers
over SciPy: explicit tolerances, typed errors, and results that say how the optimiser stopped.

```bash
pip install engg-math-optimization
```

```python
from engg_math.optimization import minimize, least_squares, linprog

minimize(lambda v: (v[0] - 1) ** 2 + (v[1] + 2) ** 2, [0.0, 0.0]).x  # [1, -2]
linprog([-1, -2], A_ub=[[1, 1], [1, 0]], b_ub=[4, 3]).x  # [0, 4]
```

## Guarantees

| Guarantee | How |
|---|---|
| No silent failure | Non-convergence raises `ConvergenceError`; `raise_on_failure=False` returns a result with `converged=False` |
| Infeasible/unbounded programs are errors | `InfeasibleError`, `UnboundedError` |
| Gradients | Analytic if given, otherwise central differences from `engg-math-numerical-computation` |
| Parameter uncertainty | `least_squares` returns `(J^T J)^-1` from the Jacobian's SVD, or `None` if rank deficient |

## API

`minimize_scalar`, `minimize` (`bfgs`, `lbfgs`, `nelder-mead`), `least_squares`, `linprog`;
results `ScalarMinimum`, `Minimum`, `LeastSquaresFit`, `LinearProgramSolution`;
errors `ConvergenceError`, `InfeasibleError`, `UnboundedError`, `DomainError`.

Constrained nonlinear programming and units are not part of 0.1.

## Development

```bash
uv sync && uv run pytest && uv run ruff check . && uv run mypy && uv run lint-imports
```
