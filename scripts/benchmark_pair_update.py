
import torch
import time
from src.classical.evoformer import PairUpdate

def benchmark_pair_update():
    B, L, D_MODEL, D_PAIR, D_HIDDEN = 8, 128, 128, 64, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s = torch.randn(B, L, D_MODEL).to(device)
    pair = torch.randn(B, L, L, D_PAIR).to(device)

    model = PairUpdate(D_MODEL, D_PAIR, D_HIDDEN).to(device)

    # Warmup
    for _ in range(5):
        _ = model(s, pair)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    n_iters = 20
    for _ in range(n_iters):
        _ = model(s, pair)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    avg_time = (end - start) / n_iters
    print(f"Average time for PairUpdate (L={L}): {avg_time:.4f}s")

    # Test with larger L
    L = 256
    s = torch.randn(B, L, D_MODEL).to(device)
    pair = torch.randn(B, L, L, D_PAIR).to(device)

    # Warmup
    for _ in range(2):
        _ = model(s, pair)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(n_iters):
        _ = model(s, pair)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    avg_time = (end - start) / n_iters
    print(f"Average time for PairUpdate (L={L}): {avg_time:.4f}s")

if __name__ == "__main__":
    benchmark_pair_update()
