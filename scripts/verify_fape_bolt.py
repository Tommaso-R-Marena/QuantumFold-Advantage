
import torch
import time
from src.training.losses import FAPELoss

def reference_fape(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask=None):
    B, L, A, _ = pred_coords.shape
    n_frames = min(L, 32)
    frame_idx = torch.linspace(0, L - 1, n_frames, device=pred_coords.device).long()

    loss_sum = 0.0
    for f in frame_idx:
        R_pred = pred_rotations[:, f] # (B, 3, 3)
        t_pred = pred_translations[:, f] # (B, 3)
        R_true = true_rotations[:, f]
        t_true = true_translations[:, f]

        # pred_local = R_pred.T @ (pred_coords - t_pred)
        # (B, 3, 3).T @ (B, L, A, 3) -> (B, L, A, 3)
        p_diff = pred_coords - t_pred.view(B, 1, 1, 3)
        p_local = torch.einsum('bij,blaj->blai', R_pred.transpose(-1, -2), p_diff)

        t_diff = true_coords - t_true.view(B, 1, 1, 3)
        t_local = torch.einsum('bij,blaj->blai', R_true.transpose(-1, -2), t_diff)

        dist = torch.sqrt(torch.sum((p_local - t_local)**2, dim=-1) + 1e-8)
        dist = torch.clamp(dist, max=10.0)

        if mask is not None:
            # mask is (B, L), dist is (B, L, A)
            dist = dist * mask.unsqueeze(-1).float()
            loss_sum += dist.sum() / (mask.sum() * A + 1e-8)
        else:
            loss_sum += dist.mean()

    return loss_sum / n_frames

def verify():
    B, L, A = 2, 64, 4
    pred_coords = torch.randn(B, L, A, 3)
    true_coords = torch.randn(B, L, A, 3)
    pred_rotations = torch.eye(3).view(1, 1, 3, 3).repeat(B, L, 1, 1)
    true_rotations = torch.eye(3).view(1, 1, 3, 3).repeat(B, L, 1, 1)
    pred_translations = torch.randn(B, L, 3)
    true_translations = torch.randn(B, L, 3)
    mask = torch.ones(B, L).bool()

    fape_mod = FAPELoss()

    out_optimized = fape_mod(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)
    out_ref = reference_fape(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)

    diff = torch.abs(out_optimized - out_ref).item()
    print(f"Difference: {diff:.2e}")
    assert diff < 1e-5

    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = fape_mod(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)
    end = time.time()
    print(f"Optimized time: {end - start:.4f}s")

    start = time.time()
    for _ in range(100):
        _ = reference_fape(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)
    end = time.time()
    print(f"Reference time: {end - start:.4f}s")

if __name__ == "__main__":
    verify()
