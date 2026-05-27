
import torch
import time
import torch.nn as nn
from torch import Tensor

class FAPELossLoop(nn.Module):
    def __init__(self, d_clamp: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.d_clamp = d_clamp
        self.eps = eps

    def forward(
        self,
        pred_coords: Tensor,
        true_coords: Tensor,
        pred_rotations: Tensor,
        true_rotations: Tensor,
        pred_translations: Tensor,
        true_translations: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        B, L, A, _ = pred_coords.shape
        n_frames = min(L, 32)
        frame_idx = torch.linspace(0, L - 1, n_frames, device=pred_coords.device).long()
        total_loss = torch.tensor(0.0, device=pred_coords.device)
        count = 0
        for fi in frame_idx:
            R_pred = pred_rotations[:, fi]
            t_pred = pred_translations[:, fi]
            R_true = true_rotations[:, fi]
            t_true = true_translations[:, fi]
            pred_local = torch.einsum(
                "bij,blaj->blai",
                R_pred.transpose(-1, -2),
                pred_coords - t_pred.unsqueeze(1).unsqueeze(2),
            )
            true_local = torch.einsum(
                "bij,blaj->blai",
                R_true.transpose(-1, -2),
                true_coords - t_true.unsqueeze(1).unsqueeze(2),
            )
            dist = torch.sqrt(
                torch.sum((pred_local - true_local) ** 2, dim=-1) + self.eps
            )
            dist = torch.clamp(dist, max=self.d_clamp)
            if mask is not None:
                dist = dist * mask.unsqueeze(-1).float()
                total_loss = total_loss + dist.sum() / (mask.sum() * A + self.eps)
            else:
                total_loss = total_loss + dist.mean()
            count += 1
        return total_loss / max(count, 1)

class FAPELossVectorized(nn.Module):
    def __init__(self, d_clamp: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.d_clamp = d_clamp
        self.eps = eps

    def forward(
        self,
        pred_coords: Tensor,
        true_coords: Tensor,
        pred_rotations: Tensor,
        true_rotations: Tensor,
        pred_translations: Tensor,
        true_translations: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        B, L, A, _ = pred_coords.shape
        n_frames = min(L, 32)
        frame_idx = torch.linspace(0, L - 1, n_frames, device=pred_coords.device).long()

        # Select frames
        R_pred = pred_rotations[:, frame_idx] # (B, F, 3, 3)
        t_pred = pred_translations[:, frame_idx] # (B, F, 3)
        R_true = true_rotations[:, frame_idx]
        t_true = true_translations[:, frame_idx]

        # Transform atoms into each residue's local frame
        # (B, 1, L, A, 3) - (B, F, 1, 1, 3) -> (B, F, L, A, 3)
        pred_rel = pred_coords.unsqueeze(1) - t_pred.unsqueeze(2).unsqueeze(3)
        true_rel = true_coords.unsqueeze(1) - t_true.unsqueeze(2).unsqueeze(3)

        # (B, F, 3, 3) @ (B, F, L, A, 3) -> (B, F, L, A, 3)
        pred_local = torch.einsum("bfji,bflaj->bflai", R_pred, pred_rel)
        true_local = torch.einsum("bfji,bflaj->bflai", R_true, true_rel)

        # Per-atom distance, clamped
        dist = torch.sqrt(
            torch.sum((pred_local - true_local) ** 2, dim=-1) + self.eps
        )
        dist = torch.clamp(dist, max=self.d_clamp)

        if mask is not None:
            dist = dist * mask.view(B, 1, L, 1).float()
            return dist.sum() / (n_frames * mask.sum() * A + self.eps)
        else:
            return dist.mean()

def benchmark():
    B, L, A = 8, 128, 4
    device = "cpu"

    pred_coords = torch.randn(B, L, A, 3, device=device)
    true_coords = torch.randn(B, L, A, 3, device=device)
    pred_rotations = torch.randn(B, L, 3, 3, device=device)
    true_rotations = torch.randn(B, L, 3, 3, device=device)
    pred_translations = torch.randn(B, L, 3, device=device)
    true_translations = torch.randn(B, L, 3, device=device)
    mask = torch.rand(B, L, device=device) > 0.2

    fape_loop = FAPELossLoop()
    fape_vec = FAPELossVectorized()

    # Parity check
    out_loop = fape_loop(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)
    out_vec = fape_vec(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)

    diff = torch.abs(out_loop - out_vec).item()
    print(f"Parity check: diff = {diff:.2e}")

    # Timing
    n_iters = 50

    # Warmup
    for _ in range(10):
        _ = fape_loop(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)

    start = time.time()
    for _ in range(n_iters):
        _ = fape_loop(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)
    loop_time = (time.time() - start) / n_iters

    # Warmup
    for _ in range(10):
        _ = fape_vec(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)

    start = time.time()
    for _ in range(n_iters):
        _ = fape_vec(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask)
    vec_time = (time.time() - start) / n_iters

    print(f"Loop time: {loop_time*1000:.2f} ms")
    print(f"Vectorized time: {vec_time*1000:.2f} ms")
    print(f"Speedup: {loop_time/vec_time:.2f}x")

if __name__ == "__main__":
    benchmark()
