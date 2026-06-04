
import torch
import time
from src.training.losses import FAPELoss

def benchmark_fape_loss():
    B, L, A = 8, 128, 14
    device = "cpu"
    pred_coords = torch.randn(B, L, A, 3, device=device)
    true_coords = torch.randn(B, L, A, 3, device=device)
    pred_rotations = torch.randn(B, L, 3, 3, device=device)
    true_rotations = torch.randn(B, L, 3, 3, device=device)
    pred_translations = torch.randn(B, L, 3, device=device)
    true_translations = torch.randn(B, L, 3, device=device)
    mask = torch.ones(B, L, device=device, dtype=torch.bool)

    loss_fn = FAPELoss()

    # Warmup
    for _ in range(2):
        _ = loss_fn(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)

    start = time.time()
    n_iters = 10
    for _ in range(n_iters):
        _ = loss_fn(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)
    end = time.time()

    avg_time = (end - start) / n_iters
    print(f"Average time for FAPELoss (L={L}): {avg_time:.4f}s")

if __name__ == "__main__":
    benchmark_fape_loss()
