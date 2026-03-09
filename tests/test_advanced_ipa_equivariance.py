import torch

from src.advanced_model import (
    InvariantPointAttention,
    StructureModule,
    batched_kabsch,
    quaternion_geodesic_interpolation,
    quaternion_to_rotation_matrix,
)


def test_quaternion_geodesic_interpolation_norm_preserved():
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    t = torch.tensor([0.5])
    q = quaternion_geodesic_interpolation(q0, q1, t)
    assert torch.allclose(torch.linalg.norm(q, dim=-1), torch.ones(1), atol=1e-5)


def test_batched_kabsch_recovers_transform():
    torch.manual_seed(0)
    pts = torch.randn(2, 10, 3)
    q = torch.randn(2, 4)
    r = quaternion_to_rotation_matrix(q)
    t = torch.randn(2, 1, 3)
    target = torch.einsum("bij,bnj->bni", r, pts) + t

    rec_r, rec_t = batched_kabsch(pts, target)
    aligned = torch.einsum("bij,bnj->bni", rec_r, pts) + rec_t.unsqueeze(1)
    assert torch.allclose(aligned, target, atol=1e-4)


def test_ipa_attention_visualization_shapes():
    ipa = InvariantPointAttention(c_s=32, c_z=16, c_hidden=8, n_heads=4)
    s = torch.randn(1, 20, 32)
    z = torch.randn(1, 20, 20, 16)
    trans = torch.randn(1, 20, 3)
    rot = torch.eye(3).view(1, 1, 3, 3).repeat(1, 20, 1, 1)
    out = ipa.visualize_attention_patterns(s, z, (trans, rot))
    assert out["attention_weights"].shape == (1, 20, 20, 4)


def test_structure_module_equivariance_metric_small():
    torch.manual_seed(1)
    module = StructureModule(c_s=32, c_z=16, n_layers=8, recycling_steps=1)
    s = torch.randn(1, 16, 32)
    z = torch.randn(1, 16, 16, 16)
    coords = torch.randn(1, 16, 3)
    err = module.validate_equivariance(s, z, coords)
    assert err.item() < 5.0
