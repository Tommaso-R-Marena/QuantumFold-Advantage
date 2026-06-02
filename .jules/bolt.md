## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - PairUpdate Optimization via Two-Step Contraction
**Learning:** Outer products like `torch.einsum("bid,bjc->bijdc", left, right)` followed by a large linear projection create massive intermediate tensors ($O(L^2 D^2)$) that bottleneck both memory and computation. Decomposing this into a two-step contraction using the linear layer's weights reshaped to $(D_{out}, D_{in1}, D_{in2})$ reduces complexity to $O(L^2 D_{out})$.
**Action:** Identify $O(L^2 D^2)$ outer products in Evoformer-like blocks and refactor them into sequential $O(L^2 D)$ or $O(L D^2)$ contractions.
