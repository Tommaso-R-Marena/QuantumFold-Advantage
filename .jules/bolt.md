## 2025-05-22 - [Vectorized FAPE Loss]
**Learning:** Replacing Python loops over frames in structural loss functions (like FAPE) with vectorized `torch.einsum` and broadcasting provides significant speedups (~2.8x on CPU) without sacrificing mathematical precision.
**Action:** Always look for loops over spatial dimensions or frames in structural modules and losses for vectorization opportunities.
