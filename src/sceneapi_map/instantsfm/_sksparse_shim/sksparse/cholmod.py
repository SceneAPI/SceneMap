"""Small ``sksparse.cholmod`` adapter backed by SciPy.

InstantSfM imports only ``cholesky`` and calls the returned factor as a
linear solver. The upstream ``scikit-sparse`` package needs SuiteSparse
headers on Windows, and the previous ``sksparse-minimal`` replacement can
abort the process during InstantSfM rotation averaging. SciPy's sparse LU
keeps the same callable surface without relying on that CHOLMOD binding.
"""

from __future__ import annotations

from typing import Any

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


class Factor:
    def __init__(self, matrix: Any) -> None:
        self._factor = splu(csc_matrix(matrix))

    def __call__(self, rhs: Any) -> Any:
        return self.solve_A(rhs)

    def solve_A(self, rhs: Any) -> Any:
        return self._factor.solve(rhs)


def cholesky(matrix: Any, **_: Any) -> Factor:
    return Factor(matrix)
