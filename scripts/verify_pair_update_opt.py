
import torch
import torch.nn as nn
import time

class PairUpdateOriginal(nn.Module):
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

class PairUpdateOptimized(nn.Module):
    def __init__(self, d_model: int = 128, d_pair: int = 64, d_hidden: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.left_proj = nn.Linear(d_model, d_hidden)
        self.right_proj = nn.Linear(d_model, d_hidden)
        self.out_proj = nn.Linear(d_hidden * d_hidden, d_pair)
        self.d_hidden = d_hidden
        self.d_pair = d_pair

    def forward(self, s, pair):
        h = self.norm(s)
        left = self.left_proj(h)   # (B, L, d_hidden)
        right = self.right_proj(h)  # (B, L, d_hidden)

        # Re-organized einsum:
        # out_proj(outer) = einsum('bid,bjc,pdc->bijp', left, right, weight) + bias
        # where weight is reshaped out_proj.weight (d_pair, d_hidden, d_hidden)

        weight = self.out_proj.weight.reshape(self.d_pair, self.d_hidden, self.d_hidden)

        # Step 1: (B, L, d_hidden) @ (d_pair, d_hidden, d_hidden) -> (B, L, d_pair, d_hidden)
        tmp = torch.einsum("bid,pdc->bipc", left, weight)

        # Step 2: (B, L, d_pair, d_hidden) @ (B, L, d_hidden) -> (B, L, L, d_pair)
        out = torch.einsum("bipc,bjc->bijp", tmp, right)

        if self.out_proj.bias is not None:
            out = out + self.out_proj.bias

        return pair + out

def test_equivalence():
    B, L, d_model, d_pair, d_hidden = 4, 32, 128, 64, 32
    s = torch.randn(B, L, d_model)
    pair = torch.randn(B, L, L, d_pair)

    orig = PairUpdateOriginal(d_model, d_pair, d_hidden)
    opt = PairUpdateOptimized(d_model, d_pair, d_hidden)

    # Copy weights
    opt.norm.load_state_dict(orig.norm.state_dict())
    opt.left_proj.load_state_dict(orig.left_proj.state_dict())
    opt.right_proj.load_state_dict(orig.right_proj.state_dict())
    opt.out_proj.load_state_dict(orig.out_proj.state_dict())

    with torch.no_grad():
        out_orig = orig(s, pair)
        out_opt = opt(s, pair)

    diff = torch.abs(out_orig - out_opt).max()
    print(f"Max difference: {diff.item():.2e}")
    assert diff < 1e-5

def benchmark():
    B, L, d_model, d_pair, d_hidden = 8, 128, 128, 64, 32
    s = torch.randn(B, L, d_model)
    pair = torch.randn(B, L, L, d_pair)

    orig = PairUpdateOriginal(d_model, d_pair, d_hidden)
    opt = PairUpdateOptimized(d_model, d_pair, d_hidden)

    # Warmup
    for _ in range(5):
        _ = orig(s, pair)
        _ = opt(s, pair)

    n_iters = 20

    start = time.time()
    for _ in range(n_iters):
        _ = orig(s, pair)
    orig_time = (time.time() - start) / n_iters

    start = time.time()
    for _ in range(n_iters):
        _ = opt(s, pair)
    opt_time = (time.time() - start) / n_iters

    print(f"Original average time: {orig_time:.4f}s")
    print(f"Optimized average time: {opt_time:.4f}s")
    print(f"Speedup: {orig_time / opt_time:.2f}x")

if __name__ == "__main__":
    test_equivalence()
    benchmark()
