
import torch
import time
from src.training.losses import FAPELoss

def benchmark_fape():
    B, L, A = 8, 128, 4
    device = "cpu"

    pred_coords = torch.randn(B, L, A, 3, device=device)
    true_coords = torch.randn(B, L, A, 3, device=device)
    pred_rotations = torch.randn(B, L, 3, 3, device=device)
    true_rotations = torch.randn(B, L, 3, 3, device=device)
    pred_translations = torch.randn(B, L, 3, device=device)
    true_translations = torch.randn(B, L, 3, device=device)
    mask = torch.ones(B, L, device=device).bool()

    fape = FAPELoss()

    # Warmup
    for _ in range(5):
        _ = fape(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)

    start_time = time.time()
    n_iters = 20
    for _ in range(n_iters):
        loss = fape(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)
    end_time = time.time()

    avg_time = (end_time - start_time) / n_iters
    print(f"Average FAPE execution time: {avg_time*1000:.2f} ms")
    return avg_time

if __name__ == "__main__":
    benchmark_fape()
