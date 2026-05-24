## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2026-05-24 - Sequential Contraction for Outer Products
**Learning:** Calculating a full outer product (B, L, L, d1*d2) before a linear projection is extremely memory-intensive. Reordering this into two sequential contractions (einsums) using the linear weight's reshaped form achieves a ~22x speedup and saves significant memory by avoiding the large intermediate tensor.
**Action:** When a linear layer follows an outer product, always consider decomposing it into two steps: Step 1 contracts one input with the weight, Step 2 contracts the result with the second input.
