## 2025-05-14 - Vectorizing FAPE Loss
**Learning:** Python loops over frames in loss functions (like FAPE) are a major bottleneck in structural biology models. Using `torch.einsum` and broadcasting allows for full vectorization, yielding >10x speedups.
**Action:** Always check for Python loops in PyTorch forward/loss methods and replace with vectorized operations where intermediate tensor sizes allow.

## 2025-05-14 - Surgical Fixes for Code Integrity
**Learning:** Broad cleanup of "known issues" in unrelated files can lead to regressions or be flagged in code review.
**Action:** Focus on the primary optimization task and only apply surgical fixes to other files if they block testing or verification of the main change.

## 2025-05-14 - Optimizing Invariant Point Attention (IPA)
**Learning:** Naive squared distance calculations ^2$ in IPA create massive (L^2 P D)$ intermediate tensors, often causing OOM or severe slowdowns. Using the identity $|a-b|^2 = |a|^2 + |b|^2 - 2a \cdot b$ avoids the large intermediate expansion. Additionally,  handles broadcasting implicitly, making manual  and  calls redundant and slower.
**Action:** Replace manual expansions with  broadcasting and use the vectorized distance formula for (L^2)$ attention mechanisms involving point coordinates.

## 2025-05-14 - Optimizing Invariant Point Attention (IPA)
**Learning:** Naive squared distance calculations (q-k)^2 in IPA create massive O(L^2 P D) intermediate tensors, often causing OOM or severe slowdowns. Using the identity |a-b|^2 = |a|^2 + |b|^2 - 2a.b avoids the large intermediate expansion. Additionally, torch.einsum handles broadcasting implicitly, making manual unsqueeze and expand calls redundant and slower.
**Action:** Replace manual expansions with einsum broadcasting and use the vectorized distance formula for O(L^2) attention mechanisms involving point coordinates.
