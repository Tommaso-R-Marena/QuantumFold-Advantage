
import torch
from src.training.losses import FAPELoss

def original_fape_forward(
    pred_coords,
    true_coords,
    pred_rotations,
    true_rotations,
    pred_translations,
    true_translations,
    mask=None,
    d_clamp=10.0,
    eps=1e-8
):
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
            torch.sum((pred_local - true_local) ** 2, dim=-1) + eps
        )
        dist = torch.clamp(dist, max=d_clamp)
        if mask is not None:
            dist = dist * mask.unsqueeze(-1).float()
            total_loss = total_loss + dist.sum() / (mask.sum() * A + eps)
        else:
            total_loss = total_loss + dist.mean()
        count += 1
    return total_loss / max(count, 1)

def verify_fape():
    B, L, A = 4, 64, 4
    device = torch.device("cpu")

    # Use double precision for verification
    pred_coords = torch.randn(B, L, A, 3).to(device).double()
    true_coords = torch.randn(B, L, A, 3).to(device).double()
    pred_rotations = torch.randn(B, L, 3, 3).to(device).double()
    true_rotations = torch.randn(B, L, 3, 3).to(device).double()
    pred_translations = torch.randn(B, L, 3).to(device).double()
    true_translations = torch.randn(B, L, 3).to(device).double()

    # Random mask
    mask = (torch.rand(B, L) > 0.2).to(device)

    fape_loss_module = FAPELoss().double()

    # Test without mask
    loss_orig = original_fape_forward(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations)
    loss_vec = fape_loss_module(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations)

    print(f"Loss (no mask) - Original: {loss_orig.item():.8f}, Vectorized: {loss_vec.item():.8f}")
    diff = (loss_orig - loss_vec).abs().item()
    print(f"Absolute difference: {diff:.2e}")
    assert diff < 1e-10, "FAPE loss mismatch without mask"

    # Test with mask
    loss_orig_mask = original_fape_forward(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask=mask)
    loss_vec_mask = fape_loss_module(pred_coords, true_coords, pred_rotations, true_rotations, pred_translations, true_translations, mask=mask)

    print(f"Loss (with mask) - Original: {loss_orig_mask.item():.8f}, Vectorized: {loss_vec_mask.item():.8f}")
    diff_mask = (loss_orig_mask - loss_vec_mask).abs().item()
    print(f"Absolute difference: {diff_mask:.2e}")
    assert diff_mask < 1e-10, "FAPE loss mismatch with mask"

    print("Verification successful!")

if __name__ == "__main__":
    verify_fape()
