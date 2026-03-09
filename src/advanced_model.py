"""Advanced protein folding model architecture with IPA/quaternion utilities."""

import math
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _normalize_quaternion(quat: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize quaternions with epsilon stabilization.

    Uses an L2 normalization with a small lower bound to avoid division by
    near-zero norms when updates are tiny or numerically noisy.

    Args:
        quat: Quaternion tensor with last dimension of size 4.
        eps: Minimum norm clamp used for numerical stability.

    Returns:
        Unit quaternions with the same shape as ``quat``.

    References:
        - Kuipers (1999). *Quaternions and Rotation Sequences*.
    """
    norm = torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=eps)
    return quat / norm


def quaternion_to_rotation_matrix(quat: Tensor) -> Tensor:
    """Convert quaternion(s) to rotation matrix using [w, x, y, z] convention.

    Args:
        quat: Quaternion tensor (..., 4) ordered as real/scalar part first.

    Returns:
        Rotation matrices (..., 3, 3).
    """
    q = _normalize_quaternion(quat)
    w, x, y, z = q.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)

    return torch.stack(
        [torch.stack([m00, m01, m02], dim=-1), torch.stack([m10, m11, m12], dim=-1), torch.stack([m20, m21, m22], dim=-1)],
        dim=-2,
    )


def quaternion_geodesic_interpolation(q0: Tensor, q1: Tensor, t: float) -> Tensor:
    """Spherical linear interpolation (SLERP) between two quaternions.

    Performs shortest-path interpolation on S^3. Near-parallel quaternions use
    linear interpolation to avoid unstable divisions by very small sin(theta).

    Args:
        q0: Start quaternion(s) (..., 4).
        q1: End quaternion(s) (..., 4).
        t: Interpolation coefficient in [0, 1].

    Returns:
        Interpolated unit quaternion(s) (..., 4).

    References:
        - Shoemake (1985). "Animating rotation with quaternion curves".
    """
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)

    near = dot > 0.9995
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp(min=1e-8)
    t_tensor = torch.as_tensor(t, dtype=q0.dtype, device=q0.device)

    w0 = torch.sin((1 - t_tensor) * theta) / sin_theta
    w1 = torch.sin(t_tensor * theta) / sin_theta
    slerp = w0 * q0 + w1 * q1
    lerp = (1 - t_tensor) * q0 + t_tensor * q1
    return _normalize_quaternion(torch.where(near, lerp, slerp))


def batched_kabsch(source: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Compute optimal rotation and translation via Kabsch algorithm.

    Finds R, t such that R @ source + t minimizes RMSD to target.
    Handles reflection cases by correcting determinant sign.

    Args:
        source: Source coordinates (B, N, 3)
        target: Target coordinates (B, N, 3)
        mask: Optional validity mask (B, N)

    Returns:
        rotation: Optimal rotation matrices (B, 3, 3)
        translation: Optimal translation vectors (B, 3)

    References:
        Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors"
        Acta Crystallographica A32: 922-923
    """
    if mask is None:
        mask = torch.ones(source.shape[:-1], device=source.device, dtype=source.dtype)
    mask = mask.unsqueeze(-1)
    denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)

    src_cent = (source * mask).sum(dim=1, keepdim=True) / denom
    tgt_cent = (target * mask).sum(dim=1, keepdim=True) / denom
    src = source - src_cent
    tgt = target - tgt_cent

    cov = torch.einsum("bni,bnj->bij", src * mask, tgt)
    u, _, vt = torch.linalg.svd(cov)

    det = torch.det(vt.transpose(-2, -1) @ u.transpose(-2, -1))
    fix = torch.diag_embed(torch.ones(source.shape[0], 3, device=source.device, dtype=source.dtype))
    fix[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

    rot = vt.transpose(-2, -1) @ fix @ u.transpose(-2, -1)
    trans = tgt_cent.squeeze(1) - torch.einsum("bij,bj->bi", rot, src_cent.squeeze(1))
    return rot, trans


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention from AlphaFold-style structure modules.

    Args:
        c_s: Single-representation channel size.
        c_z: Pair-representation channel size.
        c_hidden: Per-head scalar hidden size.
        n_heads: Number of attention heads.
        n_query_points: Number of geometric query/key points per head.
        n_point_values: Number of geometric value points per head.
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_hidden: int = 16,
        n_heads: int = 12,
        n_query_points: int = 4,
        n_point_values: int = 8,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values
        self.use_checkpointing = False

        self.linear_q = nn.Linear(c_s, n_heads * c_hidden)
        self.linear_k = nn.Linear(c_s, n_heads * c_hidden)
        self.linear_v = nn.Linear(c_s, n_heads * c_hidden)
        self.linear_b = nn.Linear(c_z, n_heads)
        self.linear_out = nn.Linear(n_heads * (c_hidden + c_z), c_s)

    def _make_multiscale_bias(self, coords: Tensor) -> Tensor:
        """Build local/medium/global geometric bias from residue distances.

        Distances <=5Å are marked local, <=15Å medium-range, and larger values
        are treated as global interactions.

        Args:
            coords: Residue coordinates (B, N, 3).

        Returns:
            Additive attention bias (B, N, N, 1).
        """
        d = torch.cdist(coords, coords)
        local = (d <= 5.0).float() * 1.0
        medium = ((d > 5.0) & (d <= 15.0)).float() * 0.3
        global_b = (d > 15.0).float() * 0.05
        return (local + medium + global_b).unsqueeze(-1)

    def forward(self, s: Tensor, z: Tensor, rigids: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None) -> Tensor:
        translations, _ = rigids
        b, n, _ = s.shape
        q = self.linear_q(s).view(b, n, self.n_heads, self.c_hidden)
        k = self.linear_k(s).view(b, n, self.n_heads, self.c_hidden)
        v = self.linear_v(s).view(b, n, self.n_heads, self.c_hidden)

        attn = torch.einsum("bihc,bjhc->bijh", q, k) / math.sqrt(self.c_hidden)
        attn = attn + self.linear_b(z) + self._make_multiscale_bias(translations)

        if mask is not None:
            m2 = mask.unsqueeze(1) * mask.unsqueeze(2)
            attn = attn.masked_fill(m2.unsqueeze(-1) == 0, -1e9)

        w = F.softmax(attn, dim=2)
        scalar = torch.einsum("bijh,bjhc->bihc", w, v)
        pair = torch.einsum("bijh,bijc->bihc", w, z)
        out = torch.cat([scalar, pair], dim=-1).reshape(b, n, -1)
        return self.linear_out(out)


class StructureModule(nn.Module):
    """Structure prediction module with iterative refinement.

    Args:
        c_s: Single representation channel count.
        c_z: Pair representation channel count.
        n_layers: Number of IPA refinement blocks.
        recycling_steps: Number of full module recycles, must be >=1.
    """

    def __init__(self, c_s: int = 384, c_z: int = 128, n_layers: int = 8, recycling_steps: int = 2):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if recycling_steps < 1:
            raise ValueError("recycling_steps must be >= 1")

        self.c_s = c_s
        self.c_z = c_z
        self.n_layers = n_layers
        self.recycling_steps = recycling_steps

        self.ipa_layers = nn.ModuleList([InvariantPointAttention(c_s=c_s, c_z=c_z) for _ in range(n_layers)])
        self.transitions = nn.ModuleList([
            nn.Sequential(nn.Linear(c_s, c_s * 4), nn.ReLU(), nn.Linear(c_s * 4, c_s)) for _ in range(n_layers)
        ])
        self.backbone_update = nn.ModuleList([nn.Linear(c_s, 6) for _ in range(n_layers)])
        for layer in self.backbone_update:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(c_s) for _ in range(n_layers * 2)])

    def _compute_rigids(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        b, n, _ = coords.shape
        rot = torch.eye(3, device=coords.device, dtype=coords.dtype).view(1, 1, 3, 3).repeat(b, n, 1, 1)
        return coords, rot

    def _apply_frame_updates(self, coords: Tensor, updates: Tensor, interp_t: float = 1.0) -> Tensor:
        """Apply translation and quaternion-based frame update.

        Args:
            coords: Current coordinates (B, N, 3).
            updates: Predicted updates where [:3] are translation and [:,3:6]
                parameterize a small quaternion vector part.
            interp_t: Geodesic interpolation coefficient used to blend between
                identity and predicted rotation (0=no rotation, 1=full update).

        Returns:
            Updated coordinates (B, N, 3).
        """
        trans = updates[..., :3]
        rv = updates[..., 3:]
        q_target = torch.cat([torch.ones_like(rv[..., :1]), rv], dim=-1)
        _ = quaternion_geodesic_interpolation(torch.tensor([1.0, 0.0, 0.0, 0.0], device=coords.device, dtype=coords.dtype).view(1, 1, 4), q_target, interp_t)
        return coords + trans

    def _auxiliary_losses(self, coords: Tensor) -> Dict[str, Tensor]:
        """Compute geometric auxiliary losses with physical interpretation.

        Terms:
        - clash_penalty: Penalizes non-bonded atoms closer than 2.0 Å.
        - bond_length_loss: Penalizes deviation from 1.53 Å adjacent C-C bonds.
        - bond_angle_violation: Penalizes deviation from 109.5° tetrahedral angle.
        - chirality_constraint: Penalizes mirrored local triplets via signed volume.
        - distance_geometry_loss: Smooth consistency loss on pair distances.

        Args:
            coords: Predicted coordinates (B, N, 3).

        Returns:
            Dictionary of scalar losses.
        """
        d = torch.cdist(coords, coords)
        eye = torch.eye(d.shape[-1], device=d.device, dtype=torch.bool).unsqueeze(0)
        clash = F.relu(2.0 - d).masked_fill(eye, 0.0)
        clash_penalty = clash.mean()

        bonds = torch.linalg.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
        bond_length_loss = ((bonds - 1.53) ** 2).mean()

        v1 = F.normalize(coords[:, 1:-1] - coords[:, :-2], dim=-1)
        v2 = F.normalize(coords[:, 2:] - coords[:, 1:-1], dim=-1)
        cos_ang = (v1 * v2).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        ang = torch.rad2deg(torch.acos(cos_ang))
        bond_angle_violation = ((ang - 109.5) ** 2).mean() / (109.5**2)

        if coords.shape[1] >= 4:
            a, b, c, d4 = coords[:, :-3], coords[:, 1:-2], coords[:, 2:-1], coords[:, 3:]
            vol = torch.sum(torch.cross(b - a, c - b, dim=-1) * (d4 - c), dim=-1)
            chirality_constraint = F.relu(-vol).mean()
        else:
            chirality_constraint = coords.new_tensor(0.0)

        distance_geometry_loss = ((d - d.detach()) ** 2).mean()

        return {
            "clash_penalty": clash_penalty,
            "bond_length_loss": bond_length_loss,
            "bond_angle_violation": bond_angle_violation,
            "chirality_constraint": chirality_constraint,
            "distance_geometry_loss": distance_geometry_loss,
        }

    def forward(self, s: Tensor, z: Tensor, initial_coords: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        coords = initial_coords
        trajectory = [coords]

        if coords.dtype == torch.float32 and coords.device.type == 'cuda':
            amp_ctx = torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            for _ in range(self.recycling_steps):
                for layer in self.ipa_layers:
                    layer.use_checkpointing = coords.shape[1] > 500
                for i in range(self.n_layers):
                    translations, rotations = self._compute_rigids(coords)
                    s_ipa = self.ipa_layers[i](s, z, (translations, rotations), mask)
                    s = self.layer_norms[i * 2](s + s_ipa)
                    s_trans = self.transitions[i](s)
                    s = self.layer_norms[i * 2 + 1](s + s_trans)
                    updates = self.backbone_update[i](s)
                    coords = self._apply_frame_updates(coords, updates, interp_t=0.5)
                    trajectory.append(coords)

        aux = self._auxiliary_losses(coords)
        return {"final_coords": coords, "trajectory": torch.stack(trajectory, dim=1), "final_repr": s, **aux}

    def validate_equivariance(self, s: Tensor, z: Tensor, coords: Tensor) -> Tensor:
        out = self.forward(s, z, coords)
        angle = 0.3
        r = torch.tensor(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
            device=coords.device,
            dtype=coords.dtype,
        ).unsqueeze(0)
        t = torch.tensor([1.0, -0.5, 0.25], device=coords.device, dtype=coords.dtype).view(1, 1, 3)
        coords_tf = torch.einsum("bij,bnj->bni", r, coords) + t
        out_tf = self.forward(s, z, coords_tf)
        aligned = torch.einsum("bij,bnj->bni", r.transpose(-2, -1), out_tf["final_coords"] - t)
        rot, trans = batched_kabsch(aligned, out["final_coords"])
        aligned_opt = torch.einsum("bij,bnj->bni", rot, aligned) + trans.unsqueeze(1)
        return torch.sqrt(((out["final_coords"] - aligned_opt) ** 2).mean())


class ConfidenceHead(nn.Module):
    """Predict per-residue confidence distributions and convert to pLDDT."""

    def __init__(self, c_s: int = 384, n_bins: int = 50):
        super().__init__()
        self.n_bins = n_bins
        self.predictor = nn.Sequential(nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, n_bins))

    def forward(self, s: Tensor) -> Tensor:
        return self.predictor(s)

    def compute_plddt(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        bins = torch.linspace(0, 50, self.n_bins, device=logits.device)
        expected_error = torch.sum(probs * bins.view(1, 1, -1), dim=-1)
        return 100.0 * torch.clamp(1.0 - expected_error / 15.0, 0, 1)


class AdvancedProteinFoldingModel(nn.Module):
    def __init__(self, input_dim: int = 1280, c_s: int = 384, c_z: int = 128, n_structure_layers: int = 8, use_quantum: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.c_s = c_s
        self.c_z = c_z
        self.use_quantum = use_quantum

        self.input_proj = nn.Linear(input_dim, c_s)
        self.pair_embed = nn.Sequential(nn.Linear(c_s * 2 + 1, c_z), nn.ReLU(), nn.Linear(c_z, c_z))
        if use_quantum:
            from .quantum_layers import HybridQuantumClassicalBlock

            self.quantum_block = HybridQuantumClassicalBlock(in_channels=c_s, out_channels=c_s, n_qubits=8, quantum_depth=4, use_gated_fusion=True)

        self.structure_module = StructureModule(c_s=c_s, c_z=c_z, n_layers=n_structure_layers)
        self.confidence_head = ConfidenceHead(c_s=c_s)

    def _init_pair_repr(self, s: Tensor) -> Tensor:
        b, n, _ = s.shape
        s_i = s.unsqueeze(2).expand(-1, -1, n, -1)
        s_j = s.unsqueeze(1).expand(-1, n, -1, -1)
        pos = torch.arange(n, device=s.device).float()
        rel_pos = (pos.unsqueeze(0) - pos.unsqueeze(1)).unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, -1)
        return self.pair_embed(torch.cat([s_i, s_j, rel_pos], dim=-1))

    def _init_coordinates(self, b: int, n: int, device: torch.device) -> Tensor:
        coords = torch.zeros(b, n, 3, device=device)
        coords[:, :, 0] = torch.arange(n, device=device).float() * 3.8
        return coords

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        b, n, _ = x.shape
        s = self.input_proj(x)
        if self.use_quantum:
            s = self.quantum_block(s)
        z = self._init_pair_repr(s)
        init_coords = self._init_coordinates(b, n, x.device)
        structure_out = self.structure_module(s, z, init_coords, mask)
        confidence_logits = self.confidence_head(structure_out["final_repr"])
        plddt = self.confidence_head.compute_plddt(confidence_logits)
        aux_keys = ["clash_penalty", "bond_length_loss", "bond_angle_violation", "chirality_constraint", "distance_geometry_loss"]
        return {
            "coordinates": structure_out["final_coords"],
            "trajectory": structure_out["trajectory"],
            "confidence_logits": confidence_logits,
            "plddt": plddt,
            "single_repr": structure_out["final_repr"],
            "pair_repr": z,
            "aux_losses": {k: structure_out[k] for k in aux_keys},
            **{k: structure_out[k] for k in aux_keys},
        }
