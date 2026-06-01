## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimizing PairUpdate via Two-Step Contraction
**Learning:** Computing outer products followed by linear projections (common in Evoformer-like architectures) creates a massive (L^2 \cdot d_{hidden}^2)$ intermediate tensor. This can be optimized by reordering the contraction into two steps: first contracting one representation with the weight matrix, then contracting the result with the second representation.
**Action:** Identify (N^2 \cdot D^2)$ patterns in tensor operations and use two-step contractions to reduce memory and time complexity to (N^2 \cdot D)$.
