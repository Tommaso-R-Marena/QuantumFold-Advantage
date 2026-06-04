
import torch
import time
from src.classical.evoformer import PairUpdate

def benchmark_pair_update():
    B, L, d_model, d_pair, d_hidden = 8, 128, 128, 64, 32
    s = torch.randn(B, L, d_model)
    pair = torch.randn(B, L, L, d_pair)

    layer = PairUpdate(d_model, d_pair, d_hidden)

    # Warmup
    for _ in range(5):
        _ = layer(s, pair)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(20):
        _ = layer(s, pair)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    avg_time = (end - start) / 20
    print(f"Average time for PairUpdate (L={L}): {avg_time:.4f}s")

if __name__ == "__main__":
    benchmark_pair_update()
