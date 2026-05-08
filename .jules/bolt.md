## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Vectorized FAPE in Core Training
**Learning:** Vectorizing FAPE using `torch.einsum` and broadcasting provides significant speedups (over 3x on CPU for typical batch sizes) by eliminating Python loops. While fully vectorized implementations can be memory-heavy for very large residue counts or atoms, it's highly efficient for the default 32-frame sampling used in this codebase.
**Action:** Prefer vectorized loss implementations in `src/training/losses.py` to match the optimizations in `src/advanced_training.py`.
