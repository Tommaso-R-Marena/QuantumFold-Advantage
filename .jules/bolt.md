## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimizing Invariant Point Attention (IPA)
**Learning:** For IPA operations, `torch.einsum` and broadcasting for pairwise distances are significant bottlenecks. Replacing `einsum` with `matmul` and using the squared distance expansion formula ($|a-b|^2 = |a|^2 + |b|^2 - 2a \cdot b$) yields a ~3x speedup on CPU.
**Action:** Prefer `torch.matmul` over `torch.einsum` for standard transformations and use the expansion formula for high-dimensional pairwise distances to avoid massive intermediate broadcasting tensors.
