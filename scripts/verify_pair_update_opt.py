
import torch
import torch.nn as nn
import time
from src.classical.evoformer import PairUpdate

def optimized_pair_update_forward(self, s, pair):
    h = self.norm(s)
    left = self.left_proj(h)
    right = self.right_proj(h)

    B, L, D = left.shape
    P = self.out_proj.out_features

    # Weight: (P, D*D) -> (P, D, D)
    W = self.out_proj.weight.view(P, D, D)

    # We want: out[b, i, j, p] = sum_{d, c} L[b, i, d] * R[b, j, c] * W[p, d, c]
    # intermediate[b, j, p, d] = sum_c R[b, j, c] * W[p, d, c]
    intermediate = torch.einsum("pdc,bjc->bjpd", W, right)

    # out[b, i, j, p] = sum_d L[b, i, d] * intermediate[b, j, p, d]
    out = torch.einsum("bid,bjpd->bijp", left, intermediate)

    return pair + out + self.out_proj.bias

def benchmark():
    B, L, D_MODEL, D_PAIR, D_HIDDEN = 4, 128, 128, 64, 32
    device = "cpu"

    s = torch.randn(B, L, D_MODEL).to(device)
    pair = torch.randn(B, L, L, D_PAIR).to(device)

    model = PairUpdate(D_MODEL, D_PAIR, D_HIDDEN).to(device)

    # Original is now patched, so we need to save the new one and restore the old one for comparison
    new_forward = PairUpdate.forward

    # Let's define the old forward manually for verification
    def old_forward(self, s, pair):
        h = self.norm(s)
        left = self.left_proj(h)
        right = self.right_proj(h)
        outer = torch.einsum("bid,bjc->bijdc", left, right)
        B, L, _, d1, d2 = outer.shape
        outer = outer.reshape(B, L, L, d1 * d2)
        return pair + self.out_proj(outer)

    # 128
    PairUpdate.forward = old_forward
    start = time.time()
    res_orig = model(s, pair)
    orig_time_128 = time.time() - start
    print(f"Original time (L={L}): {orig_time_128:.4f}s")

    PairUpdate.forward = new_forward
    start = time.time()
    res_opt = model(s, pair)
    opt_time_128 = time.time() - start
    print(f"Optimized time (L={L}): {opt_time_128:.4f}s")

    # Check correctness
    diff = torch.abs(res_orig - res_opt).max().item()
    print(f"Max difference: {diff}")

    # 256
    L = 256
    s = torch.randn(B, L, D_MODEL).to(device)
    pair = torch.randn(B, L, L, D_PAIR).to(device)

    PairUpdate.forward = old_forward
    start = time.time()
    _ = model(s, pair)
    orig_time_256 = time.time() - start
    print(f"Original time (L={L}): {orig_time_256:.4f}s")

    PairUpdate.forward = new_forward
    start = time.time()
    _ = model(s, pair)
    opt_time_256 = time.time() - start
    print(f"Optimized time (L={L}): {opt_time_256:.4f}s")

if __name__ == "__main__":
    benchmark()
