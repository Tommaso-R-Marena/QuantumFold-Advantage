## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimized PairUpdate contraction
**Learning:** Outer-product-mean updates often create massive intermediate tensors of shape (B, L, L, D^2). Reordering the contraction into two steps—first projecting one representation with the weights, then contracting with the second representation—reduces complexity from O(L^2 * D^2) to O(L^2 * D), yielding ~7-10x speedups.
**Action:** Use two-step einsum contractions for any outer-product-style layers in Evoformer-like architectures.
