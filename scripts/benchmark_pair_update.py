import torch
import torch.nn as nn
import time
from src.classical.evoformer import PairUpdate

def benchmark():
    B, L, D_S, D_P = 8, 128, 64, 128
    pair_update = PairUpdate(D_S, d_pair=D_P)
    s = torch.randn(B, L, D_S)
    pair = torch.randn(B, L, L, D_P)

    # Original implementation for comparison
    class OriginalPairUpdate(nn.Module):
        def __init__(self, d_model, d_pair=128, d_hidden=32):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.left_proj = nn.Linear(d_model, d_hidden)
            self.right_proj = nn.Linear(d_model, d_hidden)
            self.out_proj = nn.Linear(d_hidden * d_hidden, d_pair)
        def forward(self, s, pair):
            h = self.norm(s)
            left = self.left_proj(h)
            right = self.right_proj(h)
            # This is the O(L^2 * d_h^2) bottleneck
            outer = torch.einsum("bid,bjc->bijdc", left, right)
            outer = outer.reshape(outer.shape[0], outer.shape[1], outer.shape[2], -1)
            out = self.out_proj(outer)
            return pair + out

    original = OriginalPairUpdate(D_S, d_pair=D_P)
    # Copy weights
    original.norm.load_state_dict(pair_update.norm.state_dict())
    original.left_proj.load_state_dict(pair_update.left_proj.state_dict())
    original.right_proj.load_state_dict(pair_update.right_proj.state_dict())
    original.out_proj.load_state_dict(pair_update.out_proj.state_dict())

    # Verify consistency
    with torch.no_grad():
        out_orig = original(s, pair)
        out_opt = pair_update(s, pair)
        diff = (out_orig - out_opt).abs().max().item()
        print(f"Consistency check | Max diff: {diff:.2e}")

    # Warmup
    for _ in range(5):
        _ = pair_update(s, pair)
        _ = original(s, pair)

    n_iters = 20

    # Measure Optimized
    start = time.time()
    for _ in range(n_iters):
        _ = pair_update(s, pair)
    opt_time = (time.time() - start) / n_iters * 1000

    # Measure Original
    start = time.time()
    for _ in range(n_iters):
        _ = original(s, pair)
    orig_time = (time.time() - start) / n_iters * 1000

    print(f"B={B}, L={L}, D_S={D_S}, D_P={D_P}")
    print(f"Original:  {orig_time:.2f}ms")
    print(f"Optimized: {opt_time:.2f}ms")
    print(f"Speedup:   {orig_time/opt_time:.2f}x")

if __name__ == "__main__":
    benchmark()
