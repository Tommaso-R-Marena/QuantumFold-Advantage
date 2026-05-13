## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-14 - Avoiding Unintended Deletions in Complex Files
**Learning:** Using `replace_with_git_merge_diff` on files with duplicated or messy structures (like `src/quantum_hardware/backend_manager.py`) can lead to accidental deletion of critical logic if the search block matches in multiple places or if the replacement block is incorrectly scoped.
**Action:** Always perform a full file read after applying a patch to ensure no methods or logic blocks were inadvertently removed, especially in files known to have poor structural integrity.
