## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-06-22 - Optimizing Outer Product Mean Updates
**Learning:** In Evoformer-like architectures, the outer product of two per-residue representations followed by a linear projection can create a massive O(L^2 D^2) intermediate tensor. This can be optimized by splitting the linear projection into two steps using `torch.einsum`, reducing memory complexity to O(L^2 D + L D^2).
**Action:** Use two-step contraction for outer products in pair-representation updates to avoid memory bottlenecks and improve speed.
