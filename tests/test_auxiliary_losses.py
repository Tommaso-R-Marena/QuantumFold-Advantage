"""Tests for auxiliary geometric losses in structure module."""

import pytest
import torch

from src.advanced_model import AdvancedProteinFoldingModel


def test_auxiliary_losses_present_and_valid():
    """Verify all auxiliary losses are computed and have reasonable values."""
    model = AdvancedProteinFoldingModel(use_quantum=False)
    model.eval()

    x = torch.randn(2, 50, 1280)

    with torch.no_grad():
        output = model(x)

    aux = output["aux_losses"]

    expected_keys = [
        "clash_penalty",
        "bond_length_loss",
        "bond_angle_violation",
        "chirality_constraint",
        "distance_geometry_loss",
    ]

    for key in expected_keys:
        assert key in aux, f"Missing auxiliary loss: {key}"
        assert aux[key].ndim == 0, f"{key} should be scalar, got shape {aux[key].shape}"
        assert aux[key].item() >= 0, f"{key} = {aux[key].item()} should be non-negative"
        assert torch.isfinite(aux[key]), f"{key} = {aux[key].item()} is not finite"
        assert aux[key].item() < 1000, f"{key} = {aux[key].item()} is unreasonably large"

    print("✓ All auxiliary losses validated")
    for key, val in aux.items():
        print(f"  {key}: {val.item():.6f}")


def test_bond_length_loss_at_target():
    """Verify bond length loss is small when bonds are at target length (1.53 Å)."""
    model = AdvancedProteinFoldingModel(use_quantum=False)

    coords = torch.zeros(1, 20, 3)
    coords[0, :, 0] = torch.arange(20) * 1.53

    s = torch.randn(1, 20, model.c_s)
    z = torch.randn(1, 20, 20, model.c_z)

    with torch.no_grad():
        structure_out = model.structure_module(s, z, coords)

    bond_loss = structure_out["bond_length_loss"].item()
    assert bond_loss < 0.01, f"Bond length loss {bond_loss} should be < 0.01 for ideal bonds"
    print(f"✓ Bond length loss for ideal geometry: {bond_loss:.8f}")


def test_clash_penalty_increases_with_overlap():
    """Verify clash penalty increases when atoms overlap."""
    model = AdvancedProteinFoldingModel(use_quantum=False)

    s = torch.randn(1, 10, model.c_s)
    z = torch.randn(1, 10, 10, model.c_z)

    coords_separated = torch.zeros(1, 10, 3)
    coords_separated[0, :, 0] = torch.arange(10) * 5.0

    with torch.no_grad():
        out_sep = model.structure_module(s, z, coords_separated)
        clash_sep = out_sep["clash_penalty"].item()

    coords_overlap = torch.zeros(1, 10, 3)
    coords_overlap[0, :, 0] = torch.arange(10) * 0.5

    with torch.no_grad():
        out_over = model.structure_module(s, z, coords_overlap)
        clash_over = out_over["clash_penalty"].item()

    assert (
        clash_over > clash_sep * 5
    ), f"Clash penalty should increase: separated={clash_sep:.4f}, overlap={clash_over:.4f}"
    print(f"✓ Clash penalty: separated={clash_sep:.6f}, overlapping={clash_over:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
