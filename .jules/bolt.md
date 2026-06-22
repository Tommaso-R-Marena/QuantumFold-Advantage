## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Optimizing PairUpdate via Two-Step Contraction
**Learning:** In protein folding models, the outer product update  = (A \otimes B)W$ where , B \in \mathbb{R}^{L \times D}$ and  \in \mathbb{R}^{D^2 \times D'}$ is a major bottleneck if computed directly. The intermediate tensor has size (L^2 D^2)$. Replacing it with a two-step contraction  \cdot B$ reduces complexity and avoids the massive intermediate tensor.
**Action:** Replace single-step einsum outer products that are followed by linear projections with two-step contractions to achieve ~6x speedup and significantly lower peak memory.

## 2025-05-15 - Optimizing PairUpdate via Two-Step Contraction
**Learning:** In protein folding models, the outer product update O = (A \otimes B)W where A, B \in R^{L \times D} and W \in R^{D^2 \times D'} is a major bottleneck if computed directly. The intermediate tensor has size O(L^2 D^2). Replacing it with a two-step contraction (A \cdot W) \cdot B reduces complexity and avoids the massive intermediate tensor.
**Action:** Replace single-step einsum outer products that are followed by linear projections with two-step contractions to achieve ~6x speedup and significantly lower peak memory.
