## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2026-06-18 - Two-Step Contraction for Pair Updates
**Learning:** Computing a large outer product (e.g., (L^2 D^2)$) as an intermediate step in pair representation updates is a major memory and speed bottleneck. Breaking it into a two-step (LD^2)$ and (L^2D)$ contraction via `torch.einsum` yields ~8-10x speedups.
**Action:** Replace single-step outer products with two-step contractions when projecting to a higher-dimensional pair representation.
