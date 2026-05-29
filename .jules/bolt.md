## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimizing Outer-Product-Mean (PairUpdate) Contractions
**Learning:** In architectures like Evoformer, computing an explicit outer product before a linear projection (e.g., in PairUpdate) creates a massive $O(L^2 D^2)$ intermediate tensor. Reordering this into two sequential $O(L^2 D)$ contractions using `torch.einsum` yields substantial speedups (~10x) and significantly reduces memory pressure.
**Action:** Identify and refactor any "outer product followed by projection" patterns into two-step contractions to optimize for both time and memory.
