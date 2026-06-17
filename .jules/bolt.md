# Bolt's Performance Journal ⚡

## 2025-05-14 - [PairUpdate Optimization]
**Learning:** Replaced O(L^2 * D^2) outer product followed by a linear projection with a two-step contraction using torch.einsum. This avoids creating a massive intermediate tensor and provides significant speed/memory gains.
**Action:** Use associativity of matrix contraction to optimize outer-product updates in attention/evoformer blocks.

## 2025-05-14 - [Surgical CI Fixes]
**Learning:** CI failures on detached HEADs often stem from invalid Docker tags (leading hyphens) and corrupted notebook JSONs. Using 'sha-' prefix and surgical JSON patching is more reliable than broad regex.
**Action:** Always verify Docker tagging logic for empty branch variables and validate notebook JSON integrity before submission.
