
import torch
import time
from src.training.losses import FAPELoss

def benchmark_fape():
    B, L, A = 8, 128, 4
    device = torch.device("cpu") # Benchmarking on CPU first as per memory hint

    pred_coords = torch.randn(B, L, A, 3).to(device)
    true_coords = torch.randn(B, L, A, 3).to(device)
    pred_rotations = torch.randn(B, L, 3, 3).to(device)
    true_rotations = torch.randn(B, L, 3, 3).to(device)
    pred_translations = torch.randn(B, L, 3).to(device)
    true_translations = torch.randn(B, L, 3).to(device)

    fape_loss = FAPELoss()

    # Warmup
    for _ in range(5):
        _ = fape_loss(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations)

    start_time = time.time()
    n_iters = 20
    for _ in range(n_iters):
        _ = fape_loss(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations)
    end_time = time.time()

    avg_time = (end_time - start_time) / n_iters
    print(f"Average FAPE loss time (L={L}): {avg_time:.6f}s")

if __name__ == "__main__":
    benchmark_fape()
