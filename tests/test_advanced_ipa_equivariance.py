import torch

from src.advanced_model import StructureModule


def test_structure_module_equivariance_metric_small():
    """Test SE(3)-equivariance: structure output should be invariant under rigid transforms."""
    torch.manual_seed(1)
    module = StructureModule(c_s=32, c_z=16, n_layers=8, recycling_steps=1)
    module.eval()

    s = torch.randn(1, 16, 32)
    z = torch.randn(1, 16, 16, 16)
    coords = torch.randn(1, 16, 3)

    with torch.no_grad():
        err = module.validate_equivariance(s, z, coords)

    assert (
        err.item() < 0.5
    ), f"Equivariance error {err.item():.4f} Å is too large (expected < 0.5 Å)"
    print(f"✓ Equivariance error: {err.item():.6f} Å")
