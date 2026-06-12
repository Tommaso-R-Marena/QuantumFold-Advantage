## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimizing PairUpdate with Two-Step Contraction
**Learning:** In AlphaFold2-like architectures, the "Outer Product Mean" operation can create a massive intermediate tensor of shape (B, L, L, d_hidden^2). Replacing this with a two-step contraction (einsum) reduces memory and time complexity from O(L^2 * D^2) to O(L^2 * D + L * D^2).
**Action:** Identify O(L^2 * D^2) operations in attention or pair update blocks and decompose them into sequential contractions.
