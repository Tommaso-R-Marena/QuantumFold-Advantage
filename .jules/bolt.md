## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-14 - Two-Step Contraction for Pair Updates
**Learning:** Computing an explicit O(L^2 * D^2) outer product before a linear projection is a massive bottleneck and memory hog. Decomposing it into two O(L^2 * D) steps using the associativity of tensor contractions yields ~7x-10x speedups.
**Action:** Replace `torch.einsum("bid,bjc->bijdc")` followed by `Linear(d^2, p)` with two smaller einsums: `tmp = einsum("bid,pdc->bipc", left, weight)` and `out = einsum("bipc,bjc->bijp", tmp, right)`.
