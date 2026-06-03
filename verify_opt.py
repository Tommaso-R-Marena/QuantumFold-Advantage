import torch
import time
from src.classical.evoformer import PairUpdate

def benchmark():
    B, L, d_model, d_pair, d_hidden = 8, 128, 128, 64, 32
    s = torch.randn(B, L, d_model)
    pair = torch.randn(B, L, L, d_pair)
    layer = PairUpdate(d_model, d_pair, d_hidden)

    # Warmup
    for _ in range(5): _ = layer(s, pair)

    start = time.time()
    for _ in range(20): _ = layer(s, pair)
    end = time.time()
    print(f"Optimized PairUpdate time: {end - start:.4f}s")
    # Expecting < 1s. Original was ~5.5s

benchmark()
