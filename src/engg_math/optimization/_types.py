"""Shared type aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

# Array dtype/shape vary per call site; the alias keeps annotations short.
Array = npt.NDArray[Any]
ScalarFunction = Callable[[float], float]
ScalarField = Callable[[Array], float]
VectorFunction = Callable[[Array], Array]
GradientFunction = Callable[[Array], Array]

FloatArray = npt.NDArray[np.float64]
