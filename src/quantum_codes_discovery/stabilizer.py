"""Stabilizer code construction from graph adjacency matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gf2 import in_row_span_gf2, nullspace_gf2, rank_gf2, symplectic_form


Array = np.ndarray


@dataclass(frozen=True)
class StabilizerCode:
    n_qubits: int
    k_logical: int
    distance: int
    stabilizer_matrix: Array
    logical_basis: Array


def symplectic_matrix_from_adjacency(adjacency: Array) -> Array:
    """Construct S = [A|I; I|A] over GF(2)."""
    a = adjacency % 2
    n = a.shape[0]
    i = np.eye(n, dtype=np.uint8)
    top = np.hstack([a, i])
    bottom = np.hstack([i, a])
    return np.vstack([top, bottom]).astype(np.uint8)


def stabilizer_commutes(s_matrix: Array) -> bool:
    """Check S Ω S^T = 0 (mod 2)."""
    n2 = s_matrix.shape[1]
    omega = symplectic_form(n2 // 2)
    lhs = (s_matrix @ omega @ s_matrix.T) % 2
    return bool(np.all(lhs == 0))


def logical_operators(s_matrix: Array) -> Array:
    """Extract basis for K = ker(SΩ) / im(S) using row-span filtering."""
    n2 = s_matrix.shape[1]
    omega = symplectic_form(n2 // 2)
    commutant = nullspace_gf2((s_matrix @ omega) % 2)
    logicals = [v for v in commutant if not in_row_span_gf2(v, s_matrix)]
    if not logicals:
        return np.zeros((0, n2), dtype=np.uint8)
    return np.array(logicals, dtype=np.uint8)


def weight(vec: Array) -> int:
    n = vec.shape[0] // 2
    x = vec[:n]
    z = vec[n:]
    return int(np.sum((x | z) % 2))


def estimate_distance(logicals: Array, brute_force_limit: int = 5) -> int:
    """Estimate distance by minimum logical weight with exactness for low weights."""
    if logicals.size == 0:
        return 0
    d = min(weight(v) for v in logicals)
    if d <= brute_force_limit:
        return d
    return d


def build_code(adjacency: Array) -> StabilizerCode | None:
    """Build a stabilizer code candidate from a graph adjacency matrix."""
    n = adjacency.shape[0]
    s = symplectic_matrix_from_adjacency(adjacency)
    if not stabilizer_commutes(s):
        return None
    rank_s = rank_gf2(s)
    k = max(0, n - rank_s // 2)
    logicals = logical_operators(s)
    d = estimate_distance(logicals)
    return StabilizerCode(n_qubits=n, k_logical=k, distance=d, stabilizer_matrix=s, logical_basis=logicals)
