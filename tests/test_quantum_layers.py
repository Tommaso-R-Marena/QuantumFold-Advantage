"""Tests for quantum layers."""

import numpy as np
import pytest
import torch

try:
    from src.quantum_layers import EntanglementType, QuantumHybridLayer, QuantumLayer

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum layers not available")
class TestQuantumLayer:
    """Tests for QuantumLayer."""

    def test_initialization(self, device):
        """Test layer initialization."""
        layer = QuantumLayer(n_qubits=4, depth=2, device=device)
        assert layer.n_qubits == 4
        assert layer.depth == 2

    def test_forward_pass(self, device, sample_embeddings):
        """Test forward pass."""
        layer = QuantumLayer(n_qubits=4, depth=2, device=device)
        output = layer(sample_embeddings)
        assert output.shape == sample_embeddings.shape
        assert not torch.isnan(output).any()

    def test_different_entanglements(self, device, sample_embeddings):
        """Test different entanglement types."""
        for entanglement in [EntanglementType.LINEAR, EntanglementType.CIRCULAR]:
            layer = QuantumLayer(n_qubits=4, depth=2, entanglement=entanglement, device=device)
            output = layer(sample_embeddings)
            assert output.shape == sample_embeddings.shape


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum layers not available")
class TestQuantumHybridLayer:
    """Tests for QuantumHybridLayer."""

    def test_initialization(self, device):
        """Test hybrid layer initialization."""
        layer = QuantumHybridLayer(input_dim=128, n_qubits=4, depth=2, device=device)
        assert layer.n_qubits == 4

    def test_forward_pass(self, device, sample_embeddings):
        """Test hybrid forward pass."""
        layer = QuantumHybridLayer(
            input_dim=sample_embeddings.shape[-1], n_qubits=4, depth=2, device=device
        )
        output = layer(sample_embeddings)
        assert output.shape == sample_embeddings.shape
        assert not torch.isnan(output).any()
