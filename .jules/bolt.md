## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimizing Invariant Point Attention (IPA)
**Learning:** Calculating point distances in IPA using broadcasting creates a massive 6D tensor ((L^2 \cdot P)$) which is a bottleneck for both memory and speed. Using the squared distance expansion $|a-b|^2 = |a|^2 + |b|^2 - 2a \cdot b$ reduces memory to (L^2)$ and allows using optimized `torch.matmul`.
**Action:** Always use the squared distance expansion for pairwise point distances in structural models.
