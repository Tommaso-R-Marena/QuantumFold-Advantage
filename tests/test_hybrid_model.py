"""Tests for the QuantumFold-Advantage hybrid model and its components."""

import numpy as np
import pytest
import torch

from src.classical.attention import InvariantPointAttention, MultiHeadSelfAttention
from src.classical.evoformer import EvoformerBlock, EvoformerStack
from src.classical.structure_module import StructureModule
from src.data.hybrid_dataset import (
    ProteinStructureDataset,
    encode_sequence,
    generate_synthetic_proteins,
)
from src.evaluation.metrics import (
    compute_gdt_ha,
    compute_gdt_ts,
    compute_lddt,
    compute_rmsd,
    compute_tm_score,
    evaluate_structure,
    kabsch_align,
)
from src.evaluation.statistical_tests import (
    bootstrap_ci,
    cohens_d,
    compare_quantum_classical,
    holm_bonferroni,
    paired_bootstrap_test,
)
from src.models.quantumfold_advantage import (
    QuantumFoldAdvantage,
    create_classical_model,
    create_quantum_model,
)
from src.training.losses import CombinedLoss, DistanceMatrixLoss, FAPELoss

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model_kwargs():
    return dict(
        d_model=32,
        d_pair=16,
        n_heads=2,
        n_evoformer_blocks=1,
        n_structure_iterations=1,
        max_seq_len=32,
        n_qubits=2,
        n_circuit_layers=1,
        quantum_window_size=4,
    )


@pytest.fixture
def dummy_batch():
    B, L = 2, 16
    return {
        "aa_idx": torch.randint(0, 20, (B, L)),
        "physchem": torch.randn(B, L, 3),
        "mask": torch.ones(B, L, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Classical component tests
# ---------------------------------------------------------------------------


class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_with_mask(self):
        attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 10, 32)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 8:] = False
        out = attn(x, mask=mask)
        assert out.shape == (2, 10, 32)


class TestIPA:
    def test_output_shape(self):
        ipa = InvariantPointAttention(d_model=32, n_heads=2, n_query_points=2, n_value_points=2)
        B, L = 2, 8
        s = torch.randn(B, L, 32)
        t = torch.randn(B, L, 3)
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3).clone()
        out = ipa(s, t, R)
        assert out.shape == (B, L, 32)


class TestEvoformer:
    def test_block(self):
        block = EvoformerBlock(d_model=32, d_pair=16, n_heads=2)
        s = torch.randn(2, 8, 32)
        pair = torch.randn(2, 8, 8, 16)
        s_out, pair_out = block(s, pair)
        assert s_out.shape == s.shape
        assert pair_out.shape == pair.shape

    def test_stack(self):
        stack = EvoformerStack(n_blocks=2, d_model=32, d_pair=16, n_heads=2)
        s = torch.randn(2, 8, 32)
        pair = torch.randn(2, 8, 8, 16)
        s_out, pair_out = stack(s, pair)
        assert s_out.shape == s.shape


class TestStructureModule:
    def test_output_shapes(self):
        sm = StructureModule(
            d_model=32, n_heads=2, n_iterations=1, n_query_points=2, n_value_points=2
        )
        s = torch.randn(2, 8, 32)
        coords, rotations, translations = sm(s)
        assert coords.shape == (2, 8, 3, 3)
        assert rotations.shape == (2, 8, 3, 3)
        assert translations.shape == (2, 8, 3)


# ---------------------------------------------------------------------------
# Hybrid model tests
# ---------------------------------------------------------------------------


class TestQuantumFoldAdvantage:
    def test_quantum_model_forward(self, small_model_kwargs, dummy_batch):
        model = create_quantum_model(**small_model_kwargs)
        out = model(dummy_batch["aa_idx"], dummy_batch["physchem"], mask=dummy_batch["mask"])
        B, L = dummy_batch["aa_idx"].shape
        assert out["coords_backbone"].shape == (B, L, 3, 3)
        assert out["coords_ca"].shape == (B, L, 3)

    def test_classical_model_forward(self, small_model_kwargs, dummy_batch):
        model = create_classical_model(**small_model_kwargs)
        out = model(dummy_batch["aa_idx"], dummy_batch["physchem"], mask=dummy_batch["mask"])
        B, L = dummy_batch["aa_idx"].shape
        assert out["coords_backbone"].shape == (B, L, 3, 3)

    def test_quantum_flag_controls_components(self, small_model_kwargs):
        q = create_quantum_model(**small_model_kwargs)
        c = create_classical_model(**small_model_kwargs)
        assert q.quantum_enhancement is not None
        assert c.quantum_enhancement is None

    def test_gradient_flow(self, small_model_kwargs, dummy_batch):
        model = create_quantum_model(**small_model_kwargs)
        out = model(dummy_batch["aa_idx"], dummy_batch["physchem"], mask=dummy_batch["mask"])
        loss = out["coords_ca"].sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_parameter_counts(self, small_model_kwargs):
        q = create_quantum_model(**small_model_kwargs)
        c = create_classical_model(**small_model_kwargs)
        qc = q.count_parameters()
        cc = c.count_parameters()
        # Quantum model should have more parameters
        assert qc["total"] > cc["total"]
        assert "quantum_enhancement" in qc
        assert "quantum_enhancement" not in cc


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


class TestLosses:
    def test_fape_runs(self):
        fape = FAPELoss()
        B, L, A = 2, 8, 3
        pred_c = torch.randn(B, L, A, 3)
        true_c = torch.randn(B, L, A, 3)
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
        T = torch.zeros(B, L, 3)
        loss = fape(pred_c, true_c, R, R, T, T)
        assert loss.item() >= 0

    def test_dm_loss(self):
        dm = DistanceMatrixLoss()
        pred = torch.randn(2, 8, 3)
        true = torch.randn(2, 8, 3)
        loss = dm(pred, true)
        assert loss.item() >= 0

    def test_combined(self):
        cl = CombinedLoss()
        pred = {
            "coords_backbone": torch.randn(2, 8, 3, 3),
            "coords_ca": torch.randn(2, 8, 3),
            "rotations": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(2, 8, 3, 3),
            "translations": torch.zeros(2, 8, 3),
        }
        true_coords = torch.randn(2, 8, 3, 3)
        true_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(2, 8, 3, 3)
        true_trans = torch.zeros(2, 8, 3)
        loss = cl(pred, true_coords, true_rot, true_trans)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_kabsch_perfect(self):
        coords = np.random.randn(20, 3).astype(np.float32)
        aligned, R, t = kabsch_align(coords.copy(), coords)
        assert compute_rmsd(aligned, coords, align=False) < 1e-5

    def test_rmsd_positive(self):
        a = np.random.randn(15, 3)
        b = a + np.random.randn(15, 3) * 2
        assert compute_rmsd(a, b) > 0

    def test_tm_score_range(self):
        a = np.random.randn(30, 3).astype(np.float32)
        b = a + np.random.randn(30, 3) * 0.5
        tm = compute_tm_score(a, b)
        assert 0 < tm <= 1

    def test_gdt_ts_range(self):
        a = np.random.randn(20, 3).astype(np.float32)
        val = compute_gdt_ts(a, a)
        assert abs(val - 1.0) < 1e-5  # perfect match

    def test_gdt_ha_range(self):
        a = np.random.randn(20, 3).astype(np.float32)
        val = compute_gdt_ha(a, a)
        assert abs(val - 1.0) < 1e-5

    def test_lddt_perfect(self):
        a = np.random.randn(25, 3).astype(np.float32) * 5
        val = compute_lddt(a, a)
        assert abs(val - 1.0) < 1e-5

    def test_evaluate_structure(self):
        a = np.random.randn(20, 3).astype(np.float32)
        b = a + np.random.randn(20, 3) * 1.0
        m = evaluate_structure(a, b)
        assert set(m.keys()) == {"rmsd", "tm_score", "gdt_ts", "gdt_ha", "lddt"}


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_bootstrap_ci(self):
        data = np.random.randn(50)
        mean, lo, hi = bootstrap_ci(data)
        assert lo <= mean <= hi

    def test_paired_bootstrap(self):
        a = np.random.randn(30)
        b = a + 2  # clearly different
        p, diffs = paired_bootstrap_test(a, b)
        assert p < 0.05

    def test_cohens_d(self):
        rng = np.random.RandomState(0)
        a = rng.randn(20)
        b = rng.randn(20) + 2  # different draws, shifted
        d = cohens_d(a, b)
        assert d != 0

    def test_holm_bonferroni(self):
        # All significant
        assert all(holm_bonferroni([0.001, 0.002, 0.003]))
        # None significant
        assert not any(holm_bonferroni([0.5, 0.6, 0.7]))

    def test_compare_pipeline(self):
        q = {"rmsd": np.random.randn(20) + 5, "tm": np.random.randn(20) + 0.5}
        c = {"rmsd": np.random.randn(20) + 6, "tm": np.random.randn(20) + 0.4}
        results = compare_quantum_classical(q, c, n_bootstrap=500)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Data pipeline tests
# ---------------------------------------------------------------------------


class TestData:
    def test_encode_sequence(self):
        idx, props = encode_sequence("ACDEFG")
        assert idx.shape == (6,)
        assert props.shape == (6, 3)
        assert idx[0] == 0  # A

    def test_synthetic_generation(self):
        seqs, coords = generate_synthetic_proteins(10, min_len=5, max_len=20)
        assert len(seqs) == 10
        assert all(len(s) == len(c) for s, c in zip(seqs, coords))

    def test_dataset(self):
        seqs, coords = generate_synthetic_proteins(5, min_len=10, max_len=15)
        ds = ProteinStructureDataset(seqs, coords, max_len=20)
        item = ds[0]
        assert item["aa_idx"].shape == (20,)
        assert item["coords"].shape == (20, 3, 3)
        assert item["mask"].sum() == len(seqs[0])
