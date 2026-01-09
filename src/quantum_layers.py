"""Quantum circuit layers for protein folding prediction.

This module implements variational quantum circuits that can be integrated
into classical neural networks for hybrid quantum-classical protein structure
prediction.
"""

import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")

try:
    import pennylane as qml
except ImportError:
    raise ImportError("PennyLane is required. Install with: pip install pennylane")

from typing import Optional, List, Tuple


class QuantumCircuitLayer(nn.Module):
    """Variational quantum circuit layer for encoding protein features.
    
    This layer implements a parameterized quantum circuit that can process
    classical features through quantum operations, potentially capturing
    correlations not efficiently representable classically.
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of variational layers
        device_name: PennyLane device ('default.qubit' or 'lightning.qubit')
        rotation_gates: Sequence of rotation gates ('RX', 'RY', 'RZ')
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device_name: str = 'default.qubit',
        rotation_gates: List[str] = ['RY', 'RZ']
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation_gates = rotation_gates
        
        # Initialize quantum device
        self.dev = qml.device(device_name, wires=n_qubits)
        
        # Calculate number of parameters
        n_rotations = len(rotation_gates)
        self.n_params = n_qubits * n_rotations * n_layers + n_qubits  # Input encoding + variational
        
        # Trainable quantum parameters
        self.weights = nn.Parameter(torch.randn(self.n_params) * 0.1)
        
        # Build quantum circuit
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch')
    
    def _circuit(self, inputs: Tensor, weights: Tensor) -> List[float]:
        """Define the variational quantum circuit.
        
        Args:
            inputs: Classical input features (length n_qubits)
            weights: Trainable circuit parameters
            
        Returns:
            Expectation values from all qubits
        """
        # Input encoding layer
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Rotation gates
            for gate_name in self.rotation_gates:
                for qubit in range(self.n_qubits):
                    gate = getattr(qml, gate_name)
                    gate(weights[param_idx], wires=qubit)
                    param_idx += 1
            
            # Entangling layer (CNOT ladder)
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        
        # Final rotation layer
        for i in range(self.n_qubits):
            qml.RY(weights[param_idx], wires=i)
            param_idx += 1
        
        # Measure expectations
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through quantum circuit.
        
        Args:
            x: Input tensor of shape (batch_size, n_qubits) or (batch_size, features)
               If features > n_qubits, will be projected down
        
        Returns:
            Quantum circuit output of shape (batch_size, n_qubits)
        """
        batch_size = x.shape[0]
        
        # Project to n_qubits if needed
        if x.shape[1] > self.n_qubits:
            x = x[:, :self.n_qubits]
        elif x.shape[1] < self.n_qubits:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.n_qubits - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Normalize inputs to [-π, π] for better quantum encoding
        x = torch.tanh(x) * np.pi
        
        # Process each sample through quantum circuit
        outputs = []
        for i in range(batch_size):
            result = self.qnode(x[i], self.weights)
            outputs.append(torch.stack(result))
        
        return torch.stack(outputs)


class QuantumAttentionLayer(nn.Module):
    """Quantum-enhanced attention mechanism for protein structure.
    
    This layer uses quantum circuits to compute attention weights,
    potentially capturing long-range quantum correlations in protein
    conformations.
    
    Args:
        embed_dim: Dimension of input embeddings
        n_qubits: Number of qubits for quantum attention
        n_heads: Number of attention heads
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        n_qubits: int = 4,
        n_heads: int = 4
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        # Classical projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Quantum layer for attention weight enhancement
        self.quantum_layer = QuantumCircuitLayer(n_qubits=n_qubits, n_layers=2)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dimension reduction for quantum processing
        self.to_quantum = nn.Linear(self.head_dim, n_qubits)
        self.from_quantum = nn.Linear(n_qubits, self.head_dim)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass with quantum-enhanced attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Classical attention projections
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Classical attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Quantum enhancement of attention (process first few positions)
        # Create a copy to avoid in-place modification
        attn_weights_enhanced = attn_weights.clone()
        
        max_quantum_seq = min(seq_len, 8)  # Limit for efficiency
        for head in range(self.n_heads):
            for i in range(max_quantum_seq):
                # Extract attention vector for this position
                attn_vec = attn_weights[:, head, i, :max_quantum_seq]  # (batch, max_quantum_seq)
                
                # Project to quantum space
                quantum_input = self.to_quantum(attn_vec.unsqueeze(-1).expand(-1, -1, self.head_dim))
                quantum_input = quantum_input.mean(dim=-1)  # Average to get (batch, n_qubits)
                
                # Process through quantum circuit
                quantum_output = self.quantum_layer(quantum_input)  # (batch, n_qubits)
                
                # Apply quantum modulation (subtle enhancement)
                modulation = torch.sigmoid(quantum_output.mean(dim=-1, keepdim=True))  # (batch, 1)
                
                # Apply modulation without in-place operation
                attn_weights_enhanced[:, head, i, :max_quantum_seq] = (
                    attn_weights[:, head, i, :max_quantum_seq] * modulation
                )
        
        # Renormalize attention weights
        attn_weights_final = attn_weights_enhanced / (attn_weights_enhanced.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Apply attention to values
        output = torch.matmul(attn_weights_final, V)  # (batch, n_heads, seq_len, head_dim)
        
        # Concatenate heads
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(output)


class HybridQuantumClassicalBlock(nn.Module):
    """Hybrid block combining quantum and classical processing.
    
    This block implements a residual architecture where quantum circuits
    process features in parallel with classical convolutions, allowing
    the model to learn which representation is more effective.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_qubits: Number of qubits for quantum branch
        use_quantum: Whether to use quantum branch (for ablation studies)
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        n_qubits: int = 4,
        use_quantum: bool = True
    ):
        super().__init__()
        self.use_quantum = use_quantum
        
        # Classical branch
        self.classical_branch = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Quantum branch
        if use_quantum:
            self.quantum_proj_in = nn.Linear(in_channels, n_qubits)
            self.quantum_circuit = QuantumCircuitLayer(n_qubits=n_qubits, n_layers=3)
            self.quantum_proj_out = nn.Linear(n_qubits, out_channels)
        
        # Fusion layer
        fusion_input = out_channels * 2 if use_quantum else out_channels
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, out_channels),
            nn.LayerNorm(out_channels)
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through hybrid block.
        
        Args:
            x: Input tensor (batch_size, seq_len, in_channels) or (batch_size, in_channels)
            
        Returns:
            Output tensor with same shape as input (with out_channels)
        """
        # Handle both 2D and 3D inputs
        input_3d = len(x.shape) == 3
        if input_3d:
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x
        
        # Classical branch
        classical_out = self.classical_branch(x_flat)
        
        # Quantum branch (if enabled)
        if self.use_quantum:
            quantum_in = self.quantum_proj_in(x_flat)
            quantum_features = self.quantum_circuit(quantum_in)
            quantum_out = self.quantum_proj_out(quantum_features)
            
            # Concatenate branches
            combined = torch.cat([classical_out, quantum_out], dim=-1)
        else:
            combined = classical_out
        
        # Fusion
        fused = self.fusion(combined)
        
        # Residual connection
        residual = self.residual_proj(x_flat)
        output = fused + residual
        
        # Reshape if needed
        if input_3d:
            output = output.reshape(batch_size, seq_len, -1)
        
        return output
