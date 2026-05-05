## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-23 - Robust Vectorization with Einstein Summation
**Learning:** When vectorizing complex coordinate transformations like $R^T(x - t)$, using `torch.einsum` with explicit dimension names (e.g., `bfij,blaj->bflai`) is clearer and less error-prone than `unsqueeze` and `matmul`.
**Action:** Use `einsum` for multi-dimensional structural operations to maintain readability while achieving maximum performance.
