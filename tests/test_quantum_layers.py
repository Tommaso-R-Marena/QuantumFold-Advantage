"""Unit tests for quantum circuit layers."""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantum_layers import (
    QuantumCircuitLayer,
    QuantumAttentionLayer,
    HybridQuantumClassicalBlock
)


class TestQuantumCircuitLayer(unittest.TestCase):
    """Test QuantumCircuitLayer functionality."""
    
    def setUp(self):
        self.batch_size = 2
        self.n_qubits = 4
        self.layer = QuantumCircuitLayer(
            n_qubits=self.n_qubits,
            n_layers=2,
            device_name='default.qubit'
        )
    
    def test_initialization(self):
        """Test layer initialization."""
        self.assertEqual(self.layer.n_qubits, self.n_qubits)
        self.assertIsNotNone(self.layer.weights)
        self.assertEqual(self.layer.weights.requires_grad, True)
    
    def test_forward_pass(self):
        """Test forward pass with valid input."""
        x = torch.randn(self.batch_size, self.n_qubits)
        output = self.layer(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_qubits))
        self.assertTrue(torch.all(torch.abs(output) <= 1.0))  # Expectation values in [-1, 1]
    
    def test_gradient_flow(self):
        """Test that gradients flow through quantum layer."""
        x = torch.randn(self.batch_size, self.n_qubits, requires_grad=True)
        output = self.layer(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(self.layer.weights.grad)
        self.assertIsNotNone(x.grad)
    
    def test_input_projection(self):
        """Test automatic input dimension handling."""
        # Test with larger input
        x_large = torch.randn(self.batch_size, self.n_qubits + 2)
        output = self.layer(x_large)
        self.assertEqual(output.shape, (self.batch_size, self.n_qubits))
        
        # Test with smaller input (should pad)
        x_small = torch.randn(self.batch_size, self.n_qubits - 1)
        output = self.layer(x_small)
        self.assertEqual(output.shape, (self.batch_size, self.n_qubits))


class TestQuantumAttentionLayer(unittest.TestCase):
    """Test QuantumAttentionLayer functionality."""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.embed_dim = 64
        self.layer = QuantumAttentionLayer(
            embed_dim=self.embed_dim,
            n_qubits=4,
            n_heads=4
        )
    
    def test_initialization(self):
        """Test layer initialization."""
        self.assertEqual(self.layer.embed_dim, self.embed_dim)
        self.assertEqual(self.layer.n_heads, 4)
        self.assertIsNotNone(self.layer.quantum_layer)
    
    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output = self.layer(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_with_mask(self):
        """Test attention with mask."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        mask[:, :, :5, 5:] = 0  # Mask out some positions
        
        output = self.layer(x, mask=mask)
        self.assertEqual(output.shape, x.shape)
    
    def test_gradient_flow(self):
        """Test gradient flow through attention."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True)
        output = self.layer(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


class TestHybridQuantumClassicalBlock(unittest.TestCase):
    """Test HybridQuantumClassicalBlock functionality."""
    
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 64
        self.out_channels = 128
        self.block = HybridQuantumClassicalBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_qubits=4,
            use_quantum=True
        )
    
    def test_initialization(self):
        """Test block initialization."""
        self.assertTrue(self.block.use_quantum)
        self.assertIsNotNone(self.block.quantum_circuit)
        self.assertIsNotNone(self.block.classical_branch)
    
    def test_forward_2d(self):
        """Test forward pass with 2D input."""
        x = torch.randn(self.batch_size, self.in_channels)
        output = self.block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.out_channels))
    
    def test_forward_3d(self):
        """Test forward pass with 3D input (sequence)."""
        seq_len = 10
        x = torch.randn(self.batch_size, seq_len, self.in_channels)
        output = self.block(x)
        
        self.assertEqual(output.shape, (self.batch_size, seq_len, self.out_channels))
    
    def test_residual_connection(self):
        """Test residual connection is working."""
        # Same dimensions (identity residual)
        block_same = HybridQuantumClassicalBlock(
            in_channels=64,
            out_channels=64,
            use_quantum=False
        )
        x = torch.randn(self.batch_size, 64)
        output = block_same(x)
        
        # Output should not equal input due to transformations
        self.assertFalse(torch.allclose(output, x))
    
    def test_classical_only_mode(self):
        """Test block works with quantum layers disabled."""
        block_classical = HybridQuantumClassicalBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            use_quantum=False
        )
        x = torch.randn(self.batch_size, self.in_channels)
        output = block_classical(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.out_channels))
    
    def test_gradient_flow(self):
        """Test gradients flow through hybrid block."""
        x = torch.randn(self.batch_size, self.in_channels, requires_grad=True)
        output = self.block(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


class TestQuantumCircuitProperties(unittest.TestCase):
    """Test quantum circuit mathematical properties."""
    
    def test_output_bounds(self):
        """Test that quantum outputs are in valid range."""
        layer = QuantumCircuitLayer(n_qubits=4, n_layers=2)
        x = torch.randn(10, 4)
        output = layer(x)
        
        # Pauli-Z expectation values should be in [-1, 1]
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))
    
    def test_parameter_count(self):
        """Test quantum parameter count calculation."""
        n_qubits = 4
        n_layers = 3
        n_rotations = 2  # RY and RZ
        
        layer = QuantumCircuitLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_gates=['RY', 'RZ']
        )
        
        # Expected: n_qubits * n_rotations * n_layers + n_qubits (final layer)
        expected_params = n_qubits * n_rotations * n_layers + n_qubits
        self.assertEqual(layer.n_params, expected_params)
        self.assertEqual(layer.weights.numel(), expected_params)


if __name__ == '__main__':
    unittest.main()
