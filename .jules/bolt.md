## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-14 - [Vectorized FAPE Optimization]
**Learning:** Vectorizing the FAPE loss in protein folding models yields a significant speedup (~5x) by eliminating Python loop overhead. However, it requires careful multidimensional broadcasting (e.g., expanding (B, L, A, 3) to (B, Nf, L, A, 3)) and precise masking to ensure numerical consistency with the iterative reference. Narrowly scoped PRs that focus *only* on the optimization and its verification are critical for successful reviews, as adding unrelated "fixes" (formatting, CI cleanup) creates review noise that obscures the performance win.
**Action:** When implementing tensor-heavy optimizations, always provide a numerical consistency script (original vs. optimized) and a benchmark script in the PR to provide immediate, verifiable proof of improvement.
