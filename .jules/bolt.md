## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimizing Outer-Product-Mean (PairUpdate)
**Learning:** The outer-product-mean update in Evoformer blocks can create massive intermediate tensors of shape (B, L, L, d_hidden^2). For L=128 and d_hidden=32, this is 128*128*1024 = 16M elements per batch. Refactoring this into a two-step contraction using `torch.einsum` reduces memory and improves speed by ~5.8x.
**Action:** When seeing an outer product followed by a linear projection, always refactor into sequential contractions to avoid the (L^2 D^2)$ intermediate.
