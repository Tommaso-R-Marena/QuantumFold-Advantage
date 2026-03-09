"""Advanced protein folding model architecture.

Implements state-of-the-art components:
- Invariant Point Attention (IPA) with SE(3)-equivariant frame handling
- Quaternion-based frame updates with geodesic interpolation
- Iterative structure refinement with recycling
- Auxiliary geometric losses and distance-geometry constraints
"""

import math
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def _normalize_quaternion(q: Tensor, eps: float = 1e-8) -> Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=eps)


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert quaternion (..., 4) in [w, x, y, z] to rotation matrix (..., 3, 3)."""
    q = _normalize_quaternion(q)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quaternion_geodesic_interpolation(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """Slerp between q0 and q1 using geodesic interpolation."""
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.clamp((q0 * q1).sum(dim=-1, keepdim=True), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    t = t.unsqueeze(-1)
    near = sin_theta.abs() < 1e-6
    a = torch.where(near, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
    b = torch.where(near, t, torch.sin(t * theta) / sin_theta)
    return _normalize_quaternion(a * q0 + b * q1)


def batched_kabsch(source: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Batched Kabsch alignment from source to target.

    Args:
        source: (B, N, 3)
        target: (B, N, 3)
        mask: (B, N)
    Returns:
        rotation (B, 3, 3), translation (B, 3)
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
    fix = torch.eye(3, device=source.device, dtype=source.dtype).unsqueeze(0).repeat(source.shape[0], 1, 1)
    fix[:, -1, -1] = torch.where(det < 0, -1.0, 1.0)
    rot = vt.transpose(-2, -1) @ fix @ u.transpose(-2, -1)
    trans = tgt_cent.squeeze(1) - torch.einsum("bij,bj->bi", rot, src_cent.squeeze(1))
    return rot, trans


class InvariantPointAttention(nn.Module):
    """SE(3)-equivariant IPA layer with distance-bin and multiscale biases."""

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_hidden: int = 16,
        n_heads: int = 12,
        n_query_points: int = 4,
        n_point_values: int = 8,
        n_distance_bins: int = 64,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values
        self.use_checkpointing = use_checkpointing

        self.linear_q = nn.Linear(c_s, n_heads * c_hidden)
        self.linear_k = nn.Linear(c_s, n_heads * c_hidden)
        self.linear_v = nn.Linear(c_s, n_heads * c_hidden)

        self.linear_q_points = nn.Linear(c_s, n_heads * n_query_points * 3)
        self.linear_k_points = nn.Linear(c_s, n_heads * n_query_points * 3)
        self.linear_v_points = nn.Linear(c_s, n_heads * n_point_values * 3)

        self.linear_b = nn.Linear(c_z, n_heads)
        self.distance_bin_embedding = nn.Embedding(n_distance_bins, n_heads)
        self.register_buffer("distance_bin_edges", torch.linspace(0.0, 32.0, n_distance_bins + 1))

        self.scale_logits = nn.Parameter(torch.zeros(3, n_heads))  # local/medium/global per head
        self.head_weights = nn.Parameter(torch.ones(n_heads))

        out_dim = n_heads * (c_hidden + (n_point_values * 4) + c_z)
        self.linear_out = nn.Linear(out_dim, c_s)

    def _make_multiscale_bias(self, pair_dist: Tensor) -> Tensor:
        local = (pair_dist <= 5.0).float()
        medium = ((pair_dist > 5.0) & (pair_dist <= 15.0)).float()
        global_scale = torch.ones_like(pair_dist)
        scales = torch.stack([local, medium, global_scale], dim=-1).unsqueeze(-1)
        per_head = torch.softmax(self.scale_logits, dim=0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return (scales * per_head).sum(dim=-2)

    def _attention_core(self, s: Tensor, z: Tensor, translations: Tensor, rotations: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        bsz, n_res, _ = s.shape
        q = self.linear_q(s).view(bsz, n_res, self.n_heads, self.c_hidden)
        k = self.linear_k(s).view(bsz, n_res, self.n_heads, self.c_hidden)
        v = self.linear_v(s).view(bsz, n_res, self.n_heads, self.c_hidden)

        q_pts = self.linear_q_points(s).view(bsz, n_res, self.n_heads, self.n_query_points, 3)
        k_pts = self.linear_k_points(s).view(bsz, n_res, self.n_heads, self.n_query_points, 3)
        v_pts = self.linear_v_points(s).view(bsz, n_res, self.n_heads, self.n_point_values, 3)

        q_g = torch.einsum("bnij,bnhpj->bnhpi", rotations, q_pts) + translations[:, :, None, None, :]
        k_g = torch.einsum("bnij,bnhpj->bnhpi", rotations, k_pts) + translations[:, :, None, None, :]
        v_g = torch.einsum("bnij,bnhpj->bnhpi", rotations, v_pts) + translations[:, :, None, None, :]

        scalar_logits = torch.einsum("bnhc,bmhc->bnmh", q, k) / math.sqrt(self.c_hidden)
        point_dist = ((q_g[:, :, None] - k_g[:, None]) ** 2).sum(dim=-1)
        point_logits = -0.5 * point_dist.sum(dim=-1) * self.head_weights.view(1, 1, 1, -1)

        pair_bias = self.linear_b(z)
        pair_dist = torch.linalg.norm(translations[:, :, None, :] - translations[:, None, :, :], dim=-1)
        dist_bins = torch.bucketize(pair_dist, self.distance_bin_edges) - 1
        dist_bins = dist_bins.clamp(min=0, max=self.distance_bin_embedding.num_embeddings - 1)
        dist_bias = self.distance_bin_embedding(dist_bins)
        multiscale = self._make_multiscale_bias(pair_dist)

        logits = scalar_logits + point_logits + pair_bias + dist_bias + multiscale

        if mask is not None:
            m2d = (mask[:, :, None] * mask[:, None, :]).bool()
            logits = logits.masked_fill(~m2d.unsqueeze(-1), -1e4)

        attn = torch.softmax(logits.float(), dim=2).to(s.dtype)

        out_scalar = torch.einsum("bnmh,bmhc->bnhc", attn, v)
        out_pts_global = torch.einsum("bnmh,bmhpi->bnhpi", attn, v_g)
        out_pts_local = torch.einsum(
            "bnij,bnhpj->bnhpi",
            rotations.transpose(-2, -1),
            out_pts_global - translations[:, :, None, None, :],
        )

        out_pts_flat = out_pts_local.reshape(bsz, n_res, self.n_heads, -1)
        out_norms = torch.linalg.norm(out_pts_local, dim=-1)
        pair_feat = torch.einsum("bnmh,bnmc->bnhc", attn, z)

        features = torch.cat([out_scalar, out_pts_flat, out_norms, pair_feat], dim=-1).reshape(bsz, n_res, -1)
        return self.linear_out(features), attn

    def forward(self, s: Tensor, z: Tensor, rigids: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None) -> Tensor:
        translations, rotations = rigids
        if self.training and self.use_checkpointing and s.size(1) > 500:
            out, _ = checkpoint(lambda a, b, c, d, e: self._attention_core(a, b, c, d, e), s, z, translations, rotations, mask, use_reentrant=False)
            return out
        out, _ = self._attention_core(s, z, translations, rotations, mask)
        return out

    @torch.no_grad()
    def visualize_attention_patterns(self, s: Tensor, z: Tensor, rigids: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        out, attn = self._attention_core(s, z, rigids[0], rigids[1], mask)
        per_head = attn.mean(dim=1).permute(0, 2, 1)  # (B, H, N)
        return {"attention_weights": attn.detach().cpu(), "head_mean_attention": per_head.detach().cpu(), "output": out.detach().cpu()}


class StructureModule(nn.Module):
    """Structure module with 8-12 IPA layers and recycling refinement."""

    def __init__(self, c_s: int = 384, c_z: int = 128, n_layers: int = 8, recycling_steps: int = 2):
        super().__init__()
        if not 8 <= n_layers <= 12:
            raise ValueError("n_layers must be in [8, 12] for this module")
        self.c_s = c_s
        self.c_z = c_z
        self.n_layers = n_layers
        self.recycling_steps = recycling_steps

        self.ipa_layers = nn.ModuleList([InvariantPointAttention(c_s=c_s, c_z=c_z) for _ in range(n_layers)])
        self.transitions = nn.ModuleList(
            [nn.Sequential(nn.Linear(c_s, c_s * 4), nn.ReLU(), nn.Linear(c_s * 4, c_s)) for _ in range(n_layers)]
        )
        self.backbone_update = nn.ModuleList([nn.Linear(c_s, 7) for _ in range(n_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(c_s) for _ in range(n_layers * 2)])
        self.contact_predictor = nn.Linear(c_z, 1)

    def _compute_rigids(self, coords: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        b, n, _ = coords.shape
        trans = coords
        quats = torch.zeros(b, n, 4, device=coords.device, dtype=coords.dtype)
        quats[..., 0] = 1.0
        rot = quaternion_to_rotation_matrix(quats)
        return trans, rot, quats

    def _apply_frame_updates(self, coords: Tensor, quats: Tensor, updates: Tensor, interp_t: float = 0.5) -> Tuple[Tensor, Tensor, Tensor]:
        d_trans = updates[..., :3]
        d_quat_raw = updates[..., 3:]
        d_quat = _normalize_quaternion(d_quat_raw)
        interp = torch.full_like(d_quat[..., 0], interp_t)
        new_quat = quaternion_geodesic_interpolation(quats, d_quat, interp)
        rot = quaternion_to_rotation_matrix(new_quat)
        new_coords = coords + d_trans
        return new_coords, rot, new_quat

    def _auxiliary_losses(self, coords: Tensor, contact_probs: Tensor, mask: Optional[Tensor]) -> Dict[str, Tensor]:
        eps = 1e-6
        pair_dist = torch.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], dim=-1)
        if mask is None:
            mask_2d = torch.ones_like(pair_dist)
            m = torch.ones(coords.shape[:2], device=coords.device, dtype=coords.dtype)
        else:
            m = mask.float()
            mask_2d = (mask[:, :, None] * mask[:, None, :]).float()

        clash = F.relu(2.0 - pair_dist) * (1.0 - torch.eye(coords.size(1), device=coords.device).unsqueeze(0))
        clash_penalty = (clash * mask_2d).sum() / torch.clamp(mask_2d.sum(), min=1.0)

        bonds = torch.linalg.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
        bond_loss = (((bonds - 1.53) ** 2) * m[:, 1:]).sum() / torch.clamp(m[:, 1:].sum(), min=1.0)

        v1 = F.normalize(coords[:, 1:-1] - coords[:, :-2], dim=-1, eps=eps)
        v2 = F.normalize(coords[:, 2:] - coords[:, 1:-1], dim=-1, eps=eps)
        cosang = torch.clamp((v1 * v2).sum(dim=-1), -1.0 + 1e-4, 1.0 - 1e-4)
        angles = torch.rad2deg(torch.acos(cosang))
        angle_loss = (((angles - 109.5) ** 2) * m[:, 1:-1]).sum() / torch.clamp(m[:, 1:-1].sum(), min=1.0)

        if coords.size(1) > 3:
            a = coords[:, :-3]
            b = coords[:, 1:-2]
            c = coords[:, 2:-1]
            d = coords[:, 3:]
            chirality = torch.einsum("bni,bni->bn", torch.cross(b - a, c - b, dim=-1), d - c)
            chirality_loss = (F.relu(0.05 - chirality.abs()) * m[:, 3:]).sum() / torch.clamp(m[:, 3:].sum(), min=1.0)
        else:
            chirality_loss = coords.new_tensor(0.0)

        contact_target = torch.exp(-pair_dist / 8.0)
        geom_loss = F.mse_loss(contact_probs * mask_2d, contact_target * mask_2d)

        return {
            "clash_penalty": clash_penalty,
            "bond_length_loss": bond_loss,
            "bond_angle_violation": angle_loss,
            "chirality_constraint": chirality_loss,
            "distance_geometry_loss": geom_loss,
        }

    def validate_equivariance(self, s: Tensor, z: Tensor, coords: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        rand_q = _normalize_quaternion(torch.randn(coords.shape[0], 4, device=coords.device, dtype=coords.dtype))
        rand_r = quaternion_to_rotation_matrix(rand_q)
        rand_t = torch.randn(coords.shape[0], 1, 3, device=coords.device, dtype=coords.dtype)
        perturbed = torch.einsum("bij,bnj->bni", rand_r, coords) + rand_t

        out_ref = self.forward(s, z, coords, mask)["final_coords"]
        out_pert = self.forward(s, z, perturbed, mask)["final_coords"]
        aligned_r, aligned_t = batched_kabsch(out_pert, out_ref, mask)
        aligned = torch.einsum("bij,bnj->bni", aligned_r, out_pert) + aligned_t.unsqueeze(1)
        if mask is None:
            return (aligned - out_ref).pow(2).mean().sqrt()
        return (((aligned - out_ref).pow(2).sum(dim=-1) * mask).sum() / torch.clamp(mask.sum(), min=1.0)).sqrt()

    def forward(self, s: Tensor, z: Tensor, initial_coords: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        coords = initial_coords
        trajectory = [coords]

        if coords.dtype in (torch.float16, torch.bfloat16):
            amp_ctx = torch.autocast(device_type=coords.device.type, dtype=coords.dtype)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            for _ in range(self.recycling_steps):
                trans, rot, quat = self._compute_rigids(coords)
                for i in range(self.n_layers):
                    s_ipa = self.ipa_layers[i](s, z, (trans, rot), mask)
                    s = self.layer_norms[i * 2](s + s_ipa)
                    s = self.layer_norms[i * 2 + 1](s + self.transitions[i](s))
                    coords, rot, quat = self._apply_frame_updates(coords, quat, self.backbone_update[i](s), interp_t=0.35)
                    trans = coords
                    trajectory.append(coords)

        contact_probs = torch.sigmoid(self.contact_predictor(z)).squeeze(-1)
        aux = self._auxiliary_losses(coords.float(), contact_probs.float(), mask)
        return {
            "final_coords": coords,
            "trajectory": torch.stack(trajectory, dim=1),
            "final_repr": s,
            "contact_probs": contact_probs,
            **aux,
        }


class ConfidenceHead(nn.Module):
    def __init__(self, c_s: int = 384, n_bins: int = 50):
        super().__init__()
        self.n_bins = n_bins
        self.predictor = nn.Sequential(nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, n_bins))

    def forward(self, s: Tensor) -> Tensor:
        return self.predictor(s)

    def compute_plddt(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits.float(), dim=-1)
        bins = torch.linspace(0, 50, self.n_bins, device=logits.device)
        expected_error = torch.sum(probs * bins.view(1, 1, -1), dim=-1)
        return 100.0 * torch.clamp(1.0 - expected_error / 15.0, 0, 1)


class AdvancedProteinFoldingModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        c_s: int = 384,
        c_z: int = 128,
        n_structure_layers: int = 8,
        use_quantum: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.c_s = c_s
        self.c_z = c_z
        self.use_quantum = use_quantum

        self.input_proj = nn.Linear(input_dim, c_s)
        self.pair_embed = nn.Sequential(nn.Linear(c_s * 2 + 1, c_z), nn.ReLU(), nn.Linear(c_z, c_z))

        if use_quantum:
            from .quantum_layers import HybridQuantumClassicalBlock

            self.quantum_block = HybridQuantumClassicalBlock(
                in_channels=c_s,
                out_channels=c_s,
                n_qubits=8,
                quantum_depth=4,
                use_gated_fusion=True,
            )

        self.structure_module = StructureModule(c_s=c_s, c_z=c_z, n_layers=n_structure_layers)
        self.confidence_head = ConfidenceHead(c_s=c_s)

    def _init_pair_repr(self, s: Tensor) -> Tensor:
        bsz, n_res, _ = s.shape
        s_i = s.unsqueeze(2).expand(-1, -1, n_res, -1)
        s_j = s.unsqueeze(1).expand(-1, n_res, -1, -1)
        pos = torch.arange(n_res, device=s.device).float()
        rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, -1)
        return self.pair_embed(torch.cat([s_i, s_j, rel], dim=-1))

    def _init_coordinates(self, batch_size: int, n_res: int, device: torch.device) -> Tensor:
        coords = torch.zeros(batch_size, n_res, 3, device=device)
        coords[:, :, 0] = torch.arange(n_res, device=device).float() * 3.8
        return coords

    def profile_structure_module(self, x: Tensor, mask: Optional[Tensor] = None) -> Dict[str, float]:
        s = self.input_proj(x)
        z = self._init_pair_repr(s)
        coords = self._init_coordinates(x.shape[0], x.shape[1], x.device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
            _ = self.structure_module(s, z, coords, mask)
        key = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
        return {"top_ops": key}

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        bsz, n_res, _ = x.shape
        s = self.input_proj(x)
        if self.use_quantum:
            s = self.quantum_block(s)
        z = self._init_pair_repr(s)
        init_coords = self._init_coordinates(bsz, n_res, x.device)
        structure_out = self.structure_module(s, z, init_coords, mask)
        conf_logits = self.confidence_head(structure_out["final_repr"])
        plddt = self.confidence_head.compute_plddt(conf_logits)
        return {
            "coordinates": structure_out["final_coords"],
            "trajectory": structure_out["trajectory"],
            "confidence_logits": conf_logits,
            "plddt": plddt,
            "single_repr": structure_out["final_repr"],
            "pair_repr": z,
            "contact_probs": structure_out["contact_probs"],
            "aux_losses": {k: v for k, v in structure_out.items() if k.endswith("loss") or "penalty" in k or "constraint" in k or "violation" in k},
        }


class InterChainIPA(nn.Module):
    def __init__(self, c_s: int = 384, c_z: int = 128, n_heads: int = 12):
        super().__init__()
        self.ipa = InvariantPointAttention(c_s=c_s, c_z=c_z, n_heads=n_heads)

    def _to_chain_ranges(self, chain_breaks, n_res: int):
        br = chain_breaks.tolist() if isinstance(chain_breaks, torch.Tensor) else list(chain_breaks)
        bounds = [0] + [int(x) for x in br if 0 < int(x) < n_res] + [n_res]
        return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def forward(self, s, z, rigids, chain_breaks, mask=None):
        bsz, n_res, _ = s.shape
        chain_mask = torch.ones(bsz, n_res, n_res, dtype=torch.bool, device=s.device)
        per_batch = chain_breaks if (isinstance(chain_breaks, list) and chain_breaks and isinstance(chain_breaks[0], (list, tuple))) else [chain_breaks] * bsz
        for b in range(bsz):
            ranges = self._to_chain_ranges(per_batch[b], n_res)
            for i in range(len(ranges)):
                si, ei = ranges[i]
                for j in range(i + 2, len(ranges)):
                    sj, ej = ranges[j]
                    chain_mask[b, si:ei, sj:ej] = False
                    chain_mask[b, sj:ej, si:ei] = False
        z = z * chain_mask.unsqueeze(-1).float()
        return self.ipa(s, z, rigids, mask=mask)


class MultiChainStructureModule(nn.Module):
    def __init__(self, c_s: int = 384, c_z: int = 128, n_layers: int = 8, enable_inter_chain_attention: bool = True):
        super().__init__()
        self.enable_inter_chain_attention = enable_inter_chain_attention
        self.inter_chain_ipa = nn.ModuleList([InterChainIPA(c_s=c_s, c_z=c_z) for _ in range(n_layers)])
        self.transitions = nn.ModuleList([
            nn.Sequential(nn.Linear(c_s, c_s * 4), nn.ReLU(), nn.Linear(c_s * 4, c_s)) for _ in range(n_layers)
        ])
        self.backbone_update = nn.ModuleList([nn.Linear(c_s, 3) for _ in range(n_layers)])
        self.interface_predictor = nn.Sequential(nn.Linear(c_z, c_z // 2), nn.ReLU(), nn.Linear(c_z // 2, 1), nn.Sigmoid())

    def _compute_rigids_multichain(self, coords: Tensor):
        b, n, _ = coords.shape
        rotations = torch.eye(3, device=coords.device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
        return coords, rotations

    def forward(self, s: Tensor, z: Tensor, initial_coords: Tensor, chain_breaks, mask: Optional[Tensor] = None) -> Dict:
        coords = initial_coords
        s_updated = s
        if self.enable_inter_chain_attention:
            for i, layer in enumerate(self.inter_chain_ipa):
                rigids = self._compute_rigids_multichain(coords)
                s_updated = s_updated + layer(s_updated, z, rigids, chain_breaks=chain_breaks, mask=mask)
                s_updated = s_updated + self.transitions[i](s_updated)
                coords = coords + self.backbone_update[i](s_updated)
        interface_logits = self.interface_predictor(z).squeeze(-1)
        return {"final_coords": coords, "interface_contacts": interface_logits, "final_repr": s_updated}
