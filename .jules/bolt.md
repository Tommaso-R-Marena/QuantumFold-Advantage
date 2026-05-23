## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2026-05-23 - [Optimizing 3-tensor contractions in PairUpdate]
**Learning:** Naive 3-tensor contraction via a single `torch.einsum` or forming a large intermediate outer product (O(L^2 * D^2)) is highly inefficient. Reordering the operation into two sequential contractions (O(L^2 * D * P)) reduces peak memory and provides a >25x speedup.
**Action:** When performing outer-product-like updates followed by linear projections, always check if reordering the operations into smaller intermediate steps can avoid the "large intermediate" bottleneck.

## 2026-05-23 - [Optimization of PairUpdate Layer]
**Learning:** Reordering 3-tensor contractions (outer product + linear) into sequential pairwise operations significantly reduces memory overhead and improves speed by avoiding large intermediate tensors of shape (B, L, L, D^2). Using `.reshape()` instead of `.view()` ensures robustness with non-contiguous tensors.
**Action:** Always prefer sequential contractions for updates involving residue pair representations.
