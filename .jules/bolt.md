## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2026-05-26 - Manual Bias Handling in Contraction Optimizations
**Learning:** When refactoring a Linear layer applied to an outer product into sequential einsums, the bias of the original Linear layer is lost in the contraction and must be added back manually to maintain numerical equivalence.
**Action:** Always check for bias in projected layers when optimizing contractions and apply it explicitly if present.
