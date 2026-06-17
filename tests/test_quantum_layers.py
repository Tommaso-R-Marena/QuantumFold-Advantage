"""Tests for quantum layers."""

import importlib.util
from pathlib import Path

import pytest
import torch

from hypothesis import given, settings
from hypothesis import strategies as st

try:
    from src.quantum_layers import EntanglementType, QuantumHybridLayer, QuantumLayer

    QUANTUM_AVAILABLE = True
except Exception:
    module_path = Path(__file__).resolve().parent.parent / "src" / "quantum_layers.py"
    spec = importlib.util.spec_from_file_location("quantum_layers_fallback", module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        EntanglementType = module.EntanglementType
        QuantumHybridLayer = module.QuantumHybridLayer
        QuantumLayer = module.QuantumLayer
        QUANTUM_AVAILABLE = True
    else:
        QUANTUM_AVAILABLE = False


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum layers not available")
class TestQuantumLayer:
    """Tests for QuantumLayer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = QuantumLayer(n_qubits=4, depth=2)
        assert layer.n_qubits == 4
        assert layer.depth == 2

    def test_forward_pass(self, sample_embeddings):
        """Test forward pass."""
        layer = QuantumLayer(n_qubits=4, depth=2)
        output = layer(sample_embeddings)
        assert output.shape == sample_embeddings.shape
        assert not torch.isnan(output).any()

    def test_different_entanglements(self, sample_embeddings):
        """Test different entanglement types."""
        for entanglement in [EntanglementType.LINEAR, EntanglementType.CIRCULAR]:
            layer = QuantumLayer(n_qubits=4, depth=2, entanglement=entanglement)
            output = layer(sample_embeddings)
            assert output.shape == sample_embeddings.shape

    def test_parameter_shift_matches_backprop(self):
        layer = QuantumLayer(n_qubits=3, depth=2)
        x = torch.randn(3)
        max_diff = layer.validate_against_pennylane_gradients(x)
        assert max_diff < 2e-1

    def test_layerwise_relevance_is_normalized(self):
        layer = QuantumLayer(n_qubits=3, depth=2)
        x = torch.randn(3)
        rel = layer.compute_layerwise_relevance(x)
        assert rel["layer_relevance"].shape[0] == layer.n_layers
        assert torch.isclose(rel["layer_relevance"].sum(), torch.tensor(1.0), atol=1e-5)

    def test_dynamic_entanglement_selection(self):
        layer = QuantumLayer(n_qubits=4, depth=2)
        x = torch.randn(4)
        best, scores = layer.select_entanglement_topology(x)
        assert best in {"linear", "circular", "all_to_all"}
        assert set(scores.keys()) == {"linear", "circular", "all_to_all"}
        assert layer.entanglement == best

    def test_fubini_study_metric_positive_diagonal(self):
        layer = QuantumLayer(n_qubits=3, depth=2)
        x = torch.randn(3)
        g = layer.compute_fubini_study_metric(x)
        assert g.shape[0] == g.shape[1] == layer.total_params
        assert torch.all(torch.diag(g) > 0)

    def test_clifford_data_regression_enabled(self):
        layer = QuantumLayer(
            n_qubits=3,
            depth=2,
        )
        layer.add_noise = True
        layer.fit_clifford_data_regression(n_calibration=8)
        assert layer.error_mitigation_enabled

    @given(
        n_qubits=st.integers(min_value=2, max_value=4), depth=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=8)
    def test_property_forward_no_nan(self, n_qubits, depth):
        layer = QuantumLayer(n_qubits=n_qubits, depth=depth)
        x = torch.randn(2, n_qubits)
        y = layer(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum layers not available")
class TestQuantumHybridLayer:
    """Tests for QuantumHybridLayer."""

    def test_initialization(self):
        """Test hybrid layer initialization."""
        layer = QuantumHybridLayer(input_dim=128, n_qubits=4, depth=2)
        assert layer.n_qubits == 4

    def test_forward_pass(self, sample_embeddings):
        """Test hybrid forward pass."""
        layer = QuantumHybridLayer(input_dim=sample_embeddings.shape[-1], n_qubits=4, depth=2)
        output = layer(sample_embeddings)
        assert output.shape == sample_embeddings.shape
        assert not torch.isnan(output).any()
