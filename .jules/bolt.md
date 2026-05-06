## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-15 - Vectorizing Loss Frames
**Learning:** Even when frame-level loops are limited to a small number (e.g., 32), the Python overhead in a tight training loop is significant. Vectorizing across the frame dimension using `einsum` and broadcasting yielded a ~2.6x speedup on CPU.
**Action:** Consistently apply vectorization to all loss components that iterate over spatial or structural frames.
