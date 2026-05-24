
import torch
import torch.nn as nn
import time

def optimized_pair_update_logic(left, right, out_proj):
    # left: (B, L, d_hidden)
    # right: (B, L, d_hidden)
    # out_proj: nn.Linear(d_hidden * d_hidden, d_pair)

    B, L, d_hidden = left.shape
    d_pair = out_proj.out_features

    # out_proj weight shape: (d_pair, d_hidden * d_hidden)
    weight = out_proj.weight.view(d_pair, d_hidden, d_hidden)
    bias = out_proj.bias # (d_pair)

    # We want: out[b, i, j, p] = sum_{d, c} left[b, i, d] * right[b, j, c] * weight[p, d, c] + bias[p]

    # Step 1: intermediate[b, i, p, c] = sum_{d} left[b, i, d] * weight[p, d, c]
    # (B, L, d_hidden) @ (d_pair, d_hidden, d_hidden) -> (B, L, d_pair, d_hidden)
    # This can be done with einsum or matmul
    intermediate = torch.einsum("bid,pdc->bipc", left, weight)

    # Step 2: out[b, i, j, p] = sum_{c} intermediate[b, i, p, c] * right[b, j, c]
    # (B, L, d_pair, d_hidden) @ (B, L, d_hidden) -> (B, L, L, d_pair)
    # This is out[b, i, j, p] = sum_c intermediate[b, i, p, c] * right[b, j, c]
    out = torch.einsum("bipc,bjc->bijp", intermediate, right)

    if bias is not None:
        out = out + bias

    return out

def benchmark_comparison():
    B, L, d_model, d_pair, d_hidden = 8, 128, 128, 64, 32
    device = "cpu"

    left = torch.randn(B, L, d_hidden).to(device)
    right = torch.randn(B, L, d_hidden).to(device)
    out_proj = nn.Linear(d_hidden * d_hidden, d_pair).to(device)

    # Original
    start = time.time()
    for _ in range(10):
        outer = torch.einsum("bid,bjc->bijdc", left, right)
        outer = outer.reshape(B, L, L, d_hidden * d_hidden)
        res1 = out_proj(outer)
    end = time.time()
    print(f"Original time: {(end - start)*100:.2f} ms")

    # Optimized
    start = time.time()
    for _ in range(10):
        res2 = optimized_pair_update_logic(left, right, out_proj)
    end = time.time()
    print(f"Optimized time: {(end - start)*100:.2f} ms")

    diff = (res1 - res2).abs().max().item()
    print(f"Max difference: {diff:.2e}")

if __name__ == "__main__":
    benchmark_comparison()
