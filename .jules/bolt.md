## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-14 - Optimizing PairUpdate with Two-Step Contraction
**Learning:** The outer product in `PairUpdate` (`torch.einsum("bid,bjc->bijdc", left, right)`) creates an (L^2 D^2)$ intermediate tensor which is extremely memory-intensive and slow for large $ or $. Decomposing this into a two-step contraction using the output projection weights reduces the intermediate memory to (L^2 P)$ and provides significant speedups (~7-13x).
**Action:** Replace (L^2 D^2)$ outer products followed by linear layers with two-step contractions when the weights can be reshaped appropriately.
