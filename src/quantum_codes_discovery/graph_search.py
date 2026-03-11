"""Graph enumeration with coarse isomorphism deduplication."""

from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np


Array = np.ndarray


def _is_connected(adj: Array) -> bool:
    n = adj.shape[0]
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        neighbors = np.flatnonzero(adj[node])
        for nxt in neighbors:
            if int(nxt) not in seen:
                seen.add(int(nxt))
                stack.append(int(nxt))
    return len(seen) == n


def enumerate_connected_graphs(n: int, max_graphs: int | None = None) -> Iterator[Array]:
    """Enumerate connected candidate graphs as adjacency matrices.

    The deduplication key uses sorted degree sequence as a lightweight proxy for
    isomorphism rejection to keep local exploration tractable.
    """
    if n <= 0:
        return
    edges = list(combinations(range(n), 2))
    seen_signatures: set[tuple[int, ...]] = set()
    count = 0

    for mask in range(1, 1 << len(edges)):
        adj = np.zeros((n, n), dtype=np.uint8)
        for bit, (u, v) in enumerate(edges):
            if (mask >> bit) & 1:
                adj[u, v] = 1
                adj[v, u] = 1
        if not _is_connected(adj):
            continue

        signature = tuple(sorted(int(x) for x in adj.sum(axis=0)))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        yield adj
        count += 1
        if max_graphs is not None and count >= max_graphs:
            break
