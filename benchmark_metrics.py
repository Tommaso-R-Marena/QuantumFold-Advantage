
import torch
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from train import compute_rmsd, compute_tm_score

def compute_rmsd_torch(pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
    diff = pred_coords - true_coords
    rmsd = torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1), dim=-1))
    return rmsd

def compute_tm_score_torch(pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
    B, N, _ = pred_coords.shape
    d0 = 1.24 * (N ** (1.0 / 3.0)) - 1.8
    if d0 < 0.5: d0 = 0.5 # Match simplified version if needed, though simplified didn't have this floor

    distances = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=-1))
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    tm_score = torch.mean(scores, dim=-1)
    return tm_score

def benchmark():
    B = 32
    N = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pred = torch.randn(B, N, 3).to(device)
    true = torch.randn(B, N, 3).to(device)

    # Current implementation (NumPy + Loop)
    start_time = time.time()
    for _ in range(100):
        total_rmsd = 0
        total_tm = 0
        for i in range(B):
            p_np = pred[i].cpu().numpy()
            t_np = true[i].cpu().numpy()
            total_rmsd += compute_rmsd(p_np, t_np)
            total_tm += compute_tm_score(p_np, t_np)
        avg_rmsd = total_rmsd / B
        avg_tm = total_tm / B
    current_time = (time.time() - start_time) / 100
    print(f"Current implementation (Loop + NumPy): {current_time:.6f}s per batch")

    # Vectorized implementation (Torch)
    # Warmup
    _ = compute_rmsd_torch(pred, true)
    _ = compute_tm_score_torch(pred, true)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(100):
        rmsd_vec = compute_rmsd_torch(pred, true)
        tm_vec = compute_tm_score_torch(pred, true)
        avg_rmsd = rmsd_vec.mean().item()
        avg_tm = tm_vec.mean().item()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    vectorized_time = (time.time() - start_time) / 100
    print(f"Vectorized implementation (Torch): {vectorized_time:.6f}s per batch")

    print(f"Speedup: {current_time / vectorized_time:.2f}x")

if __name__ == "__main__":
    benchmark()
