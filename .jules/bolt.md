## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2024-05-15 - Vectorized FAPE Loss Implementation
**Learning:** Python loops in the inner training loop (like iterating over 32 frames in FAPE loss) are a major bottleneck. Vectorizing this operation using `torch.einsum` and broadcasting yielded a ~2.6x speedup on CPU. Also, automated notebook validation is sensitive to missing metadata keys which can occur if a notebook is manually edited or truncated.
**Action:** Always prefer vectorized operations over Python loops in loss functions. When fixing CI, check for YAML boolean alias pitfalls (like `on:` becoming `true:`) and verify notebook JSON integrity.
