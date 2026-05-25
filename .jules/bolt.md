## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimizing PairUpdate Contraction
**Learning:** The PairUpdate outer-product-mean update in Evoformer blocks can create massive intermediate tensors of shape (B, L, L, d_hidden^2). This can be optimized by reordering the contraction into two sequential einsums: (B, L, d_hidden) x (d_pair, d_hidden, d_hidden) -> (B, L, d_pair, d_hidden) and then (B, L, d_pair, d_hidden) x (B, L, d_hidden) -> (B, L, L, d_pair).
**Action:** Use two-step contraction for large outer products to achieve ~6x-25x speedup and significantly reduce memory overhead.
