# Bolt's Journal

## 2025-05-23 - [PairUpdate Optimization]
**Learning:** Replacing explicit O(L^2 * D^2) outer products with sequential contractions (O(L^2 * D)) using `torch.einsum` significantly reduces memory overhead and improves performance (~6x speedup on CPU). This is especially critical for protein folding models where L (sequence length) can be large.
**Action:** Always look for high-dimensional `einsum` or `outer` products followed by linear layers, as they are prime candidates for factorization.
