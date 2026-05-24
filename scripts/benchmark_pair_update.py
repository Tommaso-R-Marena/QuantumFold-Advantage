
import torch
import torch.nn as nn
import time
import numpy as np
from src.classical.evoformer import PairUpdate

def benchmark_pair_update():
    B, L, d_model, d_pair, d_hidden = 8, 128, 128, 64, 32

    device = "cpu"
    s = torch.randn(B, L, d_model).to(device)
    pair = torch.randn(B, L, L, d_pair).to(device)

    layer = PairUpdate(d_model, d_pair, d_hidden).to(device)

    # Warmup
    for _ in range(5):
        _ = layer(s, pair)

    # Measure
    n_iters = 20
    start = time.time()
    for _ in range(n_iters):
        _ = layer(s, pair)
    end = time.time()

    avg_time = (end - start) / n_iters
    print(f"Average time per forward pass: {avg_time*1000:.2f} ms")

    # Measure memory (approximate)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = layer(s, pair)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak memory: {peak_mem:.2f} MB")

if __name__ == "__main__":
    benchmark_pair_update()
