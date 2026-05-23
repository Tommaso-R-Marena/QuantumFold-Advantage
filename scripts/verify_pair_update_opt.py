
import torch
import torch.nn as nn
import time
from src.classical.evoformer import PairUpdate

def verify_final_implementation():
    B, L, d_model, d_pair, d_hidden = 4, 128, 128, 64, 32
    s = torch.randn(B, L, d_model)
    pair = torch.randn(B, L, L, d_pair)

    class ReferencePairUpdate(nn.Module):
        def __init__(self, d_model: int = 128, d_pair: int = 64, d_hidden: int = 32):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.left_proj = nn.Linear(d_model, d_hidden)
            self.right_proj = nn.Linear(d_model, d_hidden)
            self.out_proj = nn.Linear(d_hidden * d_hidden, d_pair)

        def forward(self, s, pair):
            h = self.norm(s)
            left = self.left_proj(h)
            right = self.right_proj(h)
            outer = torch.einsum("bid,bjc->bijdc", left, right)
            B, L, _, d1, d2 = outer.shape
            outer = outer.reshape(B, L, L, d1 * d2)
            return pair + self.out_proj(outer)

    ref_layer = ReferencePairUpdate(d_model, d_pair, d_hidden)
    opt_layer = PairUpdate(d_model, d_pair, d_hidden)

    opt_layer.load_state_dict(ref_layer.state_dict())

    with torch.no_grad():
        expected = ref_layer(s, pair)
        actual = opt_layer(s, pair)

    diff = torch.abs(expected - actual).max().item()
    print(f"Max difference: {diff:.2e}")

    if diff < 1e-5:
        print("Verification SUCCESS: Outputs match.")
    else:
        print("Verification FAILURE: Outputs do not match.")
        return

    # Benchmark
    n_iters = 50
    for _ in range(10): _ = ref_layer(s, pair)
    start = time.time()
    for _ in range(n_iters): _ = ref_layer(s, pair)
    t_ref = (time.time() - start) / n_iters

    for _ in range(10): _ = opt_layer(s, pair)
    start = time.time()
    for _ in range(n_iters): _ = opt_layer(s, pair)
    t_opt = (time.time() - start) / n_iters

    print(f"Reference: {t_ref*1000:.2f}ms, Optimized: {t_opt*1000:.2f}ms, Speedup: {t_ref/t_opt:.2f}x")

if __name__ == "__main__":
    verify_final_implementation()
