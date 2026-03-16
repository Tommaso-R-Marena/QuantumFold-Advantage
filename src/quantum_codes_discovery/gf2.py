"""GF(2) linear algebra helpers used by stabilizer code routines."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def rref_gf2(matrix: Array) -> tuple[Array, list[int]]:
    """Compute reduced row echelon form over GF(2)."""
    mat = (matrix.copy() % 2).astype(np.uint8)
    rows, cols = mat.shape
    pivot_cols: list[int] = []
    r = 0
    for c in range(cols):
        pivot = None
        for rr in range(r, rows):
            if mat[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            mat[[r, pivot]] = mat[[pivot, r]]
        for rr in range(rows):
            if rr != r and mat[rr, c]:
                mat[rr] ^= mat[r]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    return mat, pivot_cols


def rank_gf2(matrix: Array) -> int:
    """Return matrix rank over GF(2)."""
    _, pivots = rref_gf2(matrix)
    return len(pivots)


def nullspace_gf2(matrix: Array) -> Array:
    """Return a basis for ker(matrix) over GF(2) as rows."""
    rref, pivots = rref_gf2(matrix)
    rows, cols = rref.shape
    free_cols = [c for c in range(cols) if c not in pivots]
    if not free_cols:
        return np.zeros((0, cols), dtype=np.uint8)

    basis = []
    for free in free_cols:
        vec = np.zeros(cols, dtype=np.uint8)
        vec[free] = 1
        for r, pivot_col in enumerate(pivots):
            vec[pivot_col] = rref[r, free]
        basis.append(vec)
    return np.array(basis, dtype=np.uint8)


def in_row_span_gf2(vector: Array, matrix_rows: Array) -> bool:
    """Check whether vector is in span of matrix rows over GF(2)."""
    if matrix_rows.size == 0:
        return bool(np.all(vector % 2 == 0))
    combined = np.vstack([matrix_rows % 2, vector % 2]).astype(np.uint8)
    return rank_gf2(combined[:-1]) == rank_gf2(combined)


def symplectic_form(n: int) -> Array:
    """Return 2n x 2n canonical symplectic form Ω."""
    z = np.zeros((n, n), dtype=np.uint8)
    i = np.eye(n, dtype=np.uint8)
    top = np.hstack([z, i])
    bottom = np.hstack([i, z])
    return np.vstack([top, bottom])
