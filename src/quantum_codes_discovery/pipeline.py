"""End-to-end local discovery pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .database import init_db, insert_code, to_record
from .graph_search import enumerate_connected_graphs
from .stabilizer import build_code


@dataclass(frozen=True)
class DiscoveryStats:
    graphs_examined: int
    commuting_codes: int
    stored_codes: int


def run_discovery(min_n: int, max_n: int, max_graphs_per_n: int, db_path: str | Path) -> DiscoveryStats:
    conn = init_db(db_path)
    examined = 0
    commuting = 0
    stored = 0

    for n in range(min_n, max_n + 1):
        for adjacency in enumerate_connected_graphs(n, max_graphs=max_graphs_per_n):
            examined += 1
            code = build_code(adjacency)
            if code is None:
                continue
            commuting += 1
            insert_code(conn, to_record(adjacency, code))
            stored += 1

    conn.close()
    return DiscoveryStats(graphs_examined=examined, commuting_codes=commuting, stored_codes=stored)


if __name__ == "__main__":
    stats = run_discovery(min_n=4, max_n=6, max_graphs_per_n=64, db_path="data/discovered_codes.db")
    print(asdict(stats))
