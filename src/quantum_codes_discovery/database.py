"""SQLite persistence for discovered stabilizer codes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CodeRecord:
    n_qubits: int
    k_logical: int
    distance: int
    stabilizers: str
    logicals: str
    graph_hash: str


def init_db(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS discovered_codes (
            n_qubits INTEGER,
            k_logical INTEGER,
            distance INTEGER,
            stabilizers TEXT,
            logicals TEXT,
            graph_hash TEXT PRIMARY KEY
        )
        """
    )
    conn.commit()
    return conn


def graph_hash(adjacency: np.ndarray) -> str:
    return hashlib.sha256(adjacency.astype(np.uint8).tobytes()).hexdigest()


def to_record(adjacency: np.ndarray, code: object) -> CodeRecord:
    return CodeRecord(
        n_qubits=code.n_qubits,
        k_logical=code.k_logical,
        distance=code.distance,
        stabilizers=json.dumps(code.stabilizer_matrix.tolist()),
        logicals=json.dumps(code.logical_basis.tolist()),
        graph_hash=graph_hash(adjacency),
    )


def insert_code(conn: sqlite3.Connection, record: CodeRecord) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO discovered_codes
        (n_qubits, k_logical, distance, stabilizers, logicals, graph_hash)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            record.n_qubits,
            record.k_logical,
            record.distance,
            record.stabilizers,
            record.logicals,
            record.graph_hash,
        ),
    )
    conn.commit()
