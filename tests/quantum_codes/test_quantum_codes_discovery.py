import sqlite3

import numpy as np

from src.quantum_codes_discovery.gf2 import nullspace_gf2, rank_gf2
from src.quantum_codes_discovery.pipeline import run_discovery
from src.quantum_codes_discovery.stabilizer import (
    build_code,
    stabilizer_commutes,
    symplectic_matrix_from_adjacency,
)


def test_gf2_rank_and_nullspace_shapes():
    m = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    assert rank_gf2(m) == 2
    ns = nullspace_gf2(m)
    assert ns.shape[1] == 3


def test_stabilizer_commutation_runs_on_small_graph():
    adjacency = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    s = symplectic_matrix_from_adjacency(adjacency)
    assert isinstance(stabilizer_commutes(s), bool)


def test_pipeline_persists_codes(tmp_path):
    db_path = tmp_path / "codes.db"
    stats = run_discovery(min_n=3, max_n=3, max_graphs_per_n=3, db_path=db_path)
    assert stats.graphs_examined >= 1

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT COUNT(*) FROM discovered_codes").fetchone()[0]
    conn.close()
    assert rows == stats.stored_codes


def test_build_code_returns_dataclass_or_none():
    adjacency = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    code = build_code(adjacency)
    assert code is None or code.n_qubits == 3
