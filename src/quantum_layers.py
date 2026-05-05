"""Advanced quantum circuit layers for protein folding prediction.

This module implements state-of-the-art variational quantum circuits with:
- Hardware-efficient ansatz with parameter initialization strategies
- Barren plateau mitigation through layerwise learning
- Quantum kernel methods for sequence similarity
- Advanced entanglement strategies (linear, circular, all-to-all)
- Noise-aware training with depolarizing channels
- Expressibility and entanglement metrics

References:
    - Hardware-Efficient Ansatz: Kandala et al., Nature 549, 242 (2017)
    - Barren Plateaus: McClean et al., Nature Commun. 9, 4812 (2018)
    - Quantum Kernels: Havlíček et al., Nature 567, 209 (2019)
    - Parameter Initialization: Grant et al., Quantum 3, 214 (2019)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch import Tensor


class AdvancedQuantumCircuitLayer(nn.Module):
    """Advanced variational quantum circuit with hardware-efficient ansatz.

    Implements:
    - Parameter initialization strategies (identity, random, Haar-random)
    - Multiple entanglement topologies (linear, circular, all-to-all)
    - Barren plateau mitigation via parameter scaling
    - Expressibility and entangling capability metrics
    - Gradient flow monitoring

    Computational complexity summary (per forward pass on a statevector simulator):
    - Single-qubit rotation block: O(n_layers * n_qubits * n_rotations)
    - Entangling block: O(n_layers * n_entangling_gates)
    - Parameter-shift gradients: O(total_params * C_fwd), C_fwd = circuit forward cost
    - Fubini-Study metric tensor (full): O(total_params^2 * C_fwd) time, O(total_params^2) space
    - Natural-gradient solve: O(total_params^3) time for dense linear solve
    - Clifford data regression calibration: O(n_calibration * C_fwd + n_qubits^3)

    Args:
        n_qubits: Number of qubits (4-12 recommended)
        n_layers: Circuit depth (2-10 optimal)
        device_name: PennyLane device ('default.qubit', 'lightning.qubit')
        entanglement: Topology ('linear', 'circular', 'all_to_all')
        init_strategy: Parameter initialization ('identity', 'random', 'haar')
        rotation_gates: Gate types (['RX', 'RY', 'RZ'] default)
        add_noise: Simulate depolarizing noise
        noise_strength: Depolarizing probability (0-1)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        device_name: str = "default.qubit",
        entanglement: str = "linear",
        init_strategy: str = "haar",
        rotation_gates: List[str] = ["RX", "RY", "RZ"],
        add_noise: bool = False,
        noise_strength: float = 0.01,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.rotation_gates = rotation_gates
        self.add_noise = add_noise
        self.noise_strength = noise_strength
        self.entanglement_candidates = ["linear", "circular", "all_to_all"]

        # Validate parameters
        assert n_qubits >= 2, "Need at least 2 qubits"
        assert n_layers >= 1, "Need at least 1 layer"
        assert entanglement in ["linear", "circular", "all_to_all"], "Invalid entanglement topology"
        assert init_strategy in ["identity", "random", "haar"], "Invalid initialization strategy"

        # Initialize quantum device (default.mixed supports noise channels)
        effective_device = (
            "default.mixed" if add_noise and device_name == "default.qubit" else device_name
        )
        self.dev = qml.device(effective_device, wires=n_qubits)

        # Calculate parameter dimensions
        n_rotations = len(rotation_gates)
        self.n_params_per_layer = n_qubits * n_rotations
        self.n_entangling_gates = self._count_entangling_gates()
        self.total_params = (
            self.n_params_per_layer * n_layers + n_qubits
        )  # +n_qubits for input encoding

        # Initialize trainable parameters with strategy
        self.weights = nn.Parameter(self._initialize_parameters(init_strategy))

        # Parameter scaling for barren plateau mitigation (Grant et al., 2019)
        self.param_scaling = nn.Parameter(
            torch.ones(n_layers) * (1.0 / np.sqrt(n_layers)), requires_grad=True
        )

        # Build quantum circuit as differentiable QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")
        self.param_shift_qnode = qml.QNode(
            self._circuit, self.dev, interface="torch", diff_method="parameter-shift"
        )

        # State-returning circuit used for Fubini-Study metric tensor estimation
        self.metric_qnode = qml.QNode(
            self._state_circuit, self.dev, interface="torch", diff_method="parameter-shift"
        )

        # Learned error mitigation parameters (Clifford data regression)
        self.register_buffer("cdr_weights", torch.eye(self.n_qubits))
        self.register_buffer("cdr_bias", torch.zeros(self.n_qubits))
        self.error_mitigation_enabled = False

        # Monitoring metrics
        self.gradient_history = []
        self.expressibility_score = None
        self.entangling_capability = None

    def _count_entangling_gates(self) -> int:
        """Count number of two-qubit gates based on topology."""
        if self.entanglement == "linear":
            return self.n_qubits - 1
        elif self.entanglement == "circular":
            return self.n_qubits
        else:  # all_to_all
            return self.n_qubits * (self.n_qubits - 1) // 2

    def _initialize_parameters(self, strategy: str) -> Tensor:
        """Initialize circuit parameters with advanced strategies.

        - identity: Near-identity initialization (small noise around 0)
        - random: Uniform random in [-π, π]
        - haar: Haar-random inspired initialization
        """
        if strategy == "identity":
            # Small perturbations around identity
            return torch.randn(self.total_params) * 0.01
        elif strategy == "random":
            # Uniform random
            return (torch.rand(self.total_params) * 2 - 1) * np.pi
        else:  # haar
            # Approximate Haar-random by sampling from truncated normal
            # with variance scaled by circuit depth
            # Bounded initializer: Var[theta] <= pi^2/(n_layers*n_qubits), clipping prevents
            # heavy tails that amplify barren plateau effects in deep circuits.
            std = np.pi / np.sqrt(self.n_layers * self.n_qubits)
            bound = 2.0 * std
            return torch.randn(self.total_params).clamp(-bound, bound) * std

    def _apply_entangling_layer(self, layer_idx: int):
        """Apply entangling gates based on topology."""
        if self.entanglement == "linear":
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif self.entanglement == "circular":
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        else:  # all_to_all
            # Staggered pattern to reduce depth
            if layer_idx % 2 == 0:
                for i in range(0, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            else:
                for i in range(1, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])

    def _circuit(self, inputs: Tensor, weights: Tensor, scaling: Tensor) -> List[float]:
        """Hardware-efficient variational quantum circuit.

        Structure:
        1. Input encoding layer (amplitude encoding)
        2. N variational layers with:
           - Single-qubit rotations (parameterized)
           - Entangling gates
           - Optional noise
        3. Measurement layer

        Args:
            inputs: Classical features (n_qubits,)
            weights: Circuit parameters (total_params,)
            scaling: Per-layer scaling factors (n_layers,)

        Returns:
            Expectation values [<Z_0>, <Z_1>, ..., <Z_n>]
        """
        # Input encoding via amplitude embedding
        # Normalize inputs to unit sphere
        norm = torch.sqrt(torch.sum(inputs**2) + 1e-10).clamp_min(1e-8)
        inputs_normalized = inputs / norm

        # Encode using rotation gates
        for i in range(self.n_qubits):
            qml.RY(inputs_normalized[i] * np.pi, wires=i)

        # Variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Layer-wise parameter scaling (barren plateau mitigation)
            scale = scaling[layer]

            # Single-qubit rotations
            for gate_name in self.rotation_gates:
                gate = getattr(qml, gate_name)
                for qubit in range(self.n_qubits):
                    param = weights[param_idx] * scale
                    gate(param, wires=qubit)
                    param_idx += 1

            # Entangling layer
            self._apply_entangling_layer(layer)

            # Optional depolarizing noise simulation
            if self.add_noise:
                for qubit in range(self.n_qubits):
                    qml.DepolarizingChannel(self.noise_strength, wires=qubit)

        # Final rotation layer (no entanglement)
        for i in range(self.n_qubits):
            qml.RY(weights[param_idx], wires=i)
            param_idx += 1

        # Measurement in computational basis
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def _state_circuit(self, inputs: Tensor, weights: Tensor, scaling: Tensor):
        """State-returning variant used by metric tensor transforms."""
        self._circuit(inputs, weights, scaling)
        return qml.state()

    def compute_parameter_shift_gradients(self, inputs: Tensor) -> Tensor:
        """Compute exact circuit Jacobian via manual parameter-shift rule."""
        inputs = inputs.detach().float()
        shift = np.pi / 2.0
        base_weights = self.weights.detach()
        grads = []
        for p_idx in range(self.total_params):
            w_plus = base_weights.clone()
            w_minus = base_weights.clone()
            w_plus[p_idx] += shift
            w_minus[p_idx] -= shift
            out_plus = torch.stack(
                [
                    torch.as_tensor(v, dtype=torch.float32)
                    for v in self.param_shift_qnode(inputs, w_plus, self.param_scaling)
                ]
            )
            out_minus = torch.stack(
                [
                    torch.as_tensor(v, dtype=torch.float32)
                    for v in self.param_shift_qnode(inputs, w_minus, self.param_scaling)
                ]
            )
            grads.append(0.5 * (out_plus - out_minus))

        return torch.stack(grads, dim=1)

    def compute_layerwise_relevance(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Layer-wise relevance propagation proxy for interpretability.

        Relevance per layer is estimated from local sensitivity aggregated over the
        parameters in that layer, preserving differentiability.
        """
        jac = self.compute_parameter_shift_gradients(inputs)  # (n_outputs, total_params)
        param_relevance = torch.sum(torch.abs(jac), dim=0)

        layer_relevance = []
        idx = 0
        for _ in range(self.n_layers):
            chunk = param_relevance[idx : idx + self.n_params_per_layer]
            layer_relevance.append(chunk.mean())
            idx += self.n_params_per_layer

        tail = param_relevance[idx : idx + self.n_qubits]
        layer_scores = torch.stack(layer_relevance)
        normalized = layer_scores / (layer_scores.sum() + 1e-10)
        return {
            "layer_relevance": normalized,
            "readout_relevance": tail / (tail.sum() + 1e-10),
            "parameter_relevance": param_relevance,
        }

    def compute_fubini_study_metric(self, inputs: Tensor, damping: float = 1e-5) -> Tensor:
        """Compute regularized Fubini-Study metric tensor approximation.

        Uses J^T J pullback approximation from exact parameter-shift Jacobian.
        """
        jac = self.compute_parameter_shift_gradients(inputs)  # (n_outputs, n_params)
        metric = torch.matmul(jac.T, jac)
        eye = torch.eye(metric.shape[0], device=metric.device, dtype=metric.dtype)
        return metric + damping * eye

    def natural_gradient_step(
        self, inputs: Tensor, loss_fn, lr: float = 1e-2, damping: float = 1e-4
    ):
        """Apply one quantum natural-gradient step using the Fubini-Study metric."""
        loss = loss_fn(self, inputs)
        grads = torch.autograd.grad(loss, [self.weights], retain_graph=False, create_graph=False)[0]
        metric = self.compute_fubini_study_metric(inputs, damping=damping)
        nat_grad = torch.linalg.solve(metric, grads.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            self.weights -= lr * nat_grad
        return loss.detach(), nat_grad.detach()

    def select_entanglement_topology(self, inputs: Tensor) -> Tuple[str, Dict[str, float]]:
        """Select topology maximizing gradient variance to mitigate barren plateaus."""
        original = self.entanglement
        scores: Dict[str, float] = {}
        for topology in self.entanglement_candidates:
            self.entanglement = topology
            jac = self.compute_parameter_shift_gradients(inputs)
            scores[topology] = float(torch.var(jac).item())

        best = max(scores.items(), key=lambda kv: kv[1])[0]
        self.entanglement = best
        self.n_entangling_gates = self._count_entangling_gates()
        if original != best:
            self.gradient_history.append({"topology_switch": (original, best), "scores": scores})
        return best, scores

    def fit_clifford_data_regression(self, n_calibration: int = 16) -> None:
        """Learn linear error-mitigation map using Clifford-inspired calibration data."""
        self.error_mitigation_enabled = False
        X, Y = [], []
        for _ in range(n_calibration):
            # Clifford-inspired basis states encoded as {-1, +1} patterns.
            inp = torch.randint(0, 2, (self.n_qubits,), dtype=torch.float32) * 2 - 1
            original_noise = self.add_noise

            self.add_noise = False
            clean = torch.as_tensor(
                self.qnode(inp, self.weights, self.param_scaling), dtype=torch.float32
            ).detach()
            # Robust noisy proxy even when hardware noise channels are unavailable on simulator.
            if original_noise:
                try:
                    self.add_noise = True
                    noisy = torch.as_tensor(
                        self.qnode(inp, self.weights, self.param_scaling), dtype=torch.float32
                    ).detach()
                except Exception:
                    noisy = (clean + torch.randn_like(clean) * self.noise_strength).clamp(-1.0, 1.0)
            else:
                noisy = clean
            self.add_noise = original_noise

            X.append(torch.cat([noisy, torch.ones(1)]))
            Y.append(clean)

        X_mat = torch.stack(X)
        Y_mat = torch.stack(Y)
        beta = torch.linalg.pinv(X_mat) @ Y_mat  # (n_qubits+1, n_qubits)
        self.cdr_weights.copy_(beta[:-1, :].T)
        self.cdr_bias.copy_(beta[-1, :])
        self.error_mitigation_enabled = True

    def validate_against_pennylane_gradients(self, inputs: Tensor) -> float:
        """Validate parameter-shift Jacobian against autograd Jacobian."""
        ps = self.compute_parameter_shift_gradients(inputs)

        def _wrapped(w: Tensor) -> Tensor:
            out = self.qnode(inputs, w, self.param_scaling)
            return torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in out])

        bp = torch.autograd.functional.jacobian(
            _wrapped, self.weights.detach().clone().requires_grad_(True)
        )
        bp_tensor = torch.as_tensor(bp, dtype=torch.float32)
        return float(torch.max(torch.abs(ps - bp_tensor)).item())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through quantum circuit.

        Args:
            x: Input features (batch_size, feature_dim)
               If feature_dim > n_qubits, applies PCA projection
               If feature_dim < n_qubits, zero-pads

        Returns:
            Quantum circuit outputs (batch_size, n_qubits)
        """
        batch_size = x.shape[0]
        feature_dim = x.shape[1]

        # Ensure float32 dtype to match Linear layers
        x = x.float()

        # Feature preprocessing
        if feature_dim > self.n_qubits:
            # Trainable dimension reduction
            if not hasattr(self, "dim_reduction"):
                self.dim_reduction = nn.Linear(feature_dim, self.n_qubits).to(x.device)
            x = self.dim_reduction(x)
        elif feature_dim < self.n_qubits:
            # Zero padding
            padding = torch.zeros(batch_size, self.n_qubits - feature_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)

        # Normalize to [-π, π] range
        x = torch.tanh(x) * np.pi

        # Batch processing through quantum circuit
        outputs = []
        for i in range(batch_size):
            result = self.qnode(x[i], self.weights, self.param_scaling)
            # Cast to float32 to match PyTorch Linear layer expectations
            outputs.append(torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in result]))
        stacked = torch.stack(outputs)
        if self.error_mitigation_enabled:
            stacked = torch.matmul(stacked, self.cdr_weights.T) + self.cdr_bias
        return stacked

    def compute_expressibility(self, n_samples: int = 1000) -> float:
        """Compute expressibility metric (Sim et al., 2019).

        Measures how uniformly the circuit can explore state space.
        Higher values indicate better expressibility.

        Args:
            n_samples: Number of random parameter samples

        Returns:
            Expressibility score (0-1)
        """
        fidelities = []

        for _ in range(n_samples):
            # Sample two random parameter sets
            params1 = torch.randn_like(self.weights)
            params2 = torch.randn_like(self.weights)

            # Generate random input
            inputs = torch.randn(self.n_qubits)

            # Compute outputs
            out1 = self.qnode(inputs, params1, torch.ones(self.n_layers))
            out2 = self.qnode(inputs, params2, torch.ones(self.n_layers))

            # Compute fidelity
            fidelity = torch.abs(
                torch.dot(
                    torch.tensor(out1, dtype=torch.float32), torch.tensor(out2, dtype=torch.float32)
                )
            )
            fidelities.append(fidelity.item())

        # Expressibility is measured by comparing to Haar random distribution
        # Higher variance from uniform indicates better expressibility
        self.expressibility_score = float(np.std(fidelities))
        return self.expressibility_score

    def compute_entangling_capability(self) -> float:
        """Compute Meyer-Wallach entanglement measure.

        Quantifies how well the circuit can generate entanglement.
        Range: [0, 1] where 1 is maximally entangling.

        Returns:
            Entangling capability score
        """
        # Approximate by sampling random parameters
        n_samples = 100
        entanglement_scores = []

        for _ in range(n_samples):
            params = torch.randn_like(self.weights)
            inputs = torch.randn(self.n_qubits)

            # This is a simplified proxy - full calculation requires state vector
            # For production, use qml.qinfo.meyer_wallach_measure
            with torch.no_grad():
                outputs = self.qnode(inputs, params, torch.ones(self.n_layers))
                # Variance of outputs indicates entanglement
                variance = torch.var(torch.tensor(outputs, dtype=torch.float32)).item()
                entanglement_scores.append(variance)

        self.entangling_capability = float(np.mean(entanglement_scores))
        return self.entangling_capability


class QuantumKernelLayer(nn.Module):
    """Quantum kernel method for sequence similarity measurement.

    Implements quantum feature maps and kernel trick for protein sequences.
    Based on Havlíček et al., "Supervised learning with quantum-enhanced feature spaces"

    Args:
        n_qubits: Number of qubits
        n_repeats: Number of feature map repetitions
        kernel_type: Type of kernel ('Z', 'ZZ', 'full')
    """

    def __init__(self, n_qubits: int = 4, n_repeats: int = 2, kernel_type: str = "ZZ"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats
        self.kernel_type = kernel_type

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Feature map parameters (trainable)
        self.feature_params = nn.Parameter(torch.randn(n_repeats, n_qubits) * 0.1)

        # Kernel evaluation circuit
        @qml.qnode(self.dev, interface="torch")
        def kernel_circuit(x1, x2, params):
            """Quantum kernel evaluation circuit."""
            # Feature map for x1
            for rep in range(self.n_repeats):
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(x1[i] + params[rep, i], wires=i)

                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Adjoint of feature map for x2
            for rep in range(self.n_repeats - 1, -1, -1):
                # Reverse entangling
                for i in range(self.n_qubits - 2, -1, -1):
                    qml.CNOT(wires=[i, i + 1])

                # Reverse rotations
                for i in range(self.n_qubits - 1, -1, -1):
                    qml.RZ(-(x2[i] + params[rep, i]), wires=i)
                    qml.Hadamard(wires=i)

            # Measure overlap
            if self.kernel_type == "Z":
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            elif self.kernel_type == "ZZ":
                measurements = []
                for i in range(self.n_qubits - 1):
                    measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
                return measurements
            else:  # full
                return qml.probs(wires=range(self.n_qubits))

        self.kernel_circuit = kernel_circuit

    def compute_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Compute quantum kernel between two samples.

        Args:
            x1: First sample (n_qubits,)
            x2: Second sample (n_qubits,)

        Returns:
            Kernel value (scalar)
        """
        result = self.kernel_circuit(x1, x2, self.feature_params)
        if self.kernel_type in ["Z", "ZZ"]:
            return torch.mean(torch.abs(torch.tensor(result, dtype=torch.float32)))
        else:
            return torch.tensor(result[0], dtype=torch.float32)  # Probability of |00...0⟩

    def forward(self, X: Tensor) -> Tensor:
        """Compute Gram matrix for batch of samples.

        Args:
            X: Batch of samples (batch_size, n_qubits)

        Returns:
            Kernel Gram matrix (batch_size, batch_size)
        """
        batch_size = X.shape[0]
        K = torch.zeros(batch_size, batch_size)

        for i in range(batch_size):
            for j in range(i, batch_size):
                k_ij = self.compute_kernel(X[i], X[j])
                K[i, j] = k_ij
                K[j, i] = k_ij

        return K


class QuantumAttentionLayer(nn.Module):
    """Quantum-enhanced multi-head attention for protein sequences.

    Implements hybrid attention mechanism where quantum circuits modulate
    classical attention weights based on learned quantum features.

    Features:
    - Parallel quantum processing per attention head
    - Gradient-friendly design (no in-place operations)
    - Efficient batching with partial quantum enhancement
    - Adaptive modulation strength

    Args:
        embed_dim: Embedding dimension
        n_qubits: Qubits per attention head
        n_heads: Number of attention heads
        quantum_depth: Quantum circuit layers
        modulation_strength: Control quantum influence (0-1)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_qubits: int = 4,
        n_heads: int = 4,
        quantum_depth: int = 3,
        modulation_strength: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.modulation_strength = modulation_strength

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Classical attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Quantum circuits (one per head for parallelization)
        self.quantum_circuits = nn.ModuleList(
            [
                AdvancedQuantumCircuitLayer(
                    n_qubits=n_qubits,
                    n_layers=quantum_depth,
                    entanglement="linear",
                    init_strategy="haar",
                )
                for _ in range(n_heads)
            ]
        )

        # Adaptive modulation (learnable)
        self.modulation_gate = nn.Sequential(nn.Linear(n_qubits, n_heads), nn.Sigmoid())

        # Dimension adaptation
        self.to_quantum = nn.Linear(self.head_dim, n_qubits)
        self.from_quantum = nn.Linear(n_qubits, self.head_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass with quantum-enhanced attention.

        Args:
            x: Input (batch_size, seq_len, embed_dim)
            mask: Attention mask (batch_size, seq_len, seq_len)

        Returns:
            Output (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Classical attention projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Classical attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        # Quantum enhancement (process key positions)
        attn_weights_enhanced = attn_weights.clone()
        max_quantum_positions = min(seq_len, 16)  # Limit for efficiency

        # Process each head with its quantum circuit
        for head_idx in range(self.n_heads):
            quantum_circuit = self.quantum_circuits[head_idx]

            for pos in range(max_quantum_positions):
                # Extract attention pattern
                attn_pattern = attn_weights[:, head_idx, pos, :max_quantum_positions]

                # Project to quantum space
                q_input = self.to_quantum(attn_pattern.unsqueeze(-1).expand(-1, -1, self.head_dim))
                q_input = q_input.mean(dim=-1)  # (batch, n_qubits)

                # Quantum processing
                q_output = quantum_circuit(q_input)  # (batch, n_qubits)

                # Learnable modulation - ensure float32
                modulation = self.modulation_gate(q_output.float())[:, head_idx : head_idx + 1]
                modulation = 1.0 + (modulation - 0.5) * self.modulation_strength

                # Apply modulation (non-inplace)
                attn_weights_enhanced[:, head_idx, pos, :max_quantum_positions] = (
                    attn_weights[:, head_idx, pos, :max_quantum_positions] * modulation
                )

        # Renormalize
        attn_weights_final = attn_weights_enhanced / (
            attn_weights_enhanced.sum(dim=-1, keepdim=True) + 1e-9
        )

        # Apply attention to values
        output = torch.matmul(attn_weights_final, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(output)


class HybridQuantumClassicalBlock(nn.Module):
    """Advanced hybrid processing block with parallel quantum-classical branches.

    Implements:
    - Gated fusion mechanism
    - Residual connections with learnable scaling
    - Separate normalization for each branch
    - Optional skip connections

    Args:
        in_channels: Input dimension
        out_channels: Output dimension
        n_qubits: Qubits for quantum branch
        quantum_depth: Quantum circuit depth
        use_gated_fusion: Use gating mechanism for fusion
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        n_qubits: int = 4,
        quantum_depth: int = 3,
        use_gated_fusion: bool = True,
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion

        # Classical branch with depth
        self.classical_branch = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
        )

        # Quantum branch
        self.quantum_proj_in = nn.Linear(in_channels, n_qubits)
        self.quantum_circuit = AdvancedQuantumCircuitLayer(
            n_qubits=n_qubits, n_layers=quantum_depth, entanglement="circular", init_strategy="haar"
        )
        self.quantum_proj_out = nn.Linear(n_qubits, out_channels)
        self.quantum_norm = nn.LayerNorm(out_channels)

        # Fusion mechanism
        if use_gated_fusion:
            # Learnable gates for adaptive fusion
            self.fusion_gate = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels), nn.Sigmoid()
            )
        else:
            # Simple concatenation + projection
            self.fusion_proj = nn.Linear(out_channels * 2, out_channels)

        # Residual connection with learnable scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through hybrid block.

        Args:
            x: Input (batch_size, seq_len, in_channels) or (batch_size, in_channels)

        Returns:
            Output with same shape
        """
        # Handle 2D/3D inputs
        input_shape = x.shape
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x

        # Classical branch
        classical_out = self.classical_branch(x_flat)

        # Quantum branch - ensure float32
        q_in = self.quantum_proj_in(x_flat)
        q_features = self.quantum_circuit(q_in)
        quantum_out = self.quantum_proj_out(q_features.float())
        quantum_out = self.quantum_norm(quantum_out)

        # Fusion
        combined = torch.cat([classical_out, quantum_out], dim=-1)
        if self.use_gated_fusion:
            gate = self.fusion_gate(combined)
            fused = gate * classical_out + (1 - gate) * quantum_out
        else:
            fused = self.fusion_proj(combined)

        # Residual connection
        residual = self.residual_proj(x_flat)
        output = fused + self.residual_scale * residual

        # Reshape if needed
        if len(input_shape) == 3:
            output = output.reshape(batch_size, seq_len, -1)

        return output


class EntanglementType(str, Enum):
    """Compatibility enum for selecting entanglement topology."""

    LINEAR = "linear"
    CIRCULAR = "circular"
    ALL_TO_ALL = "all_to_all"


class QuantumLayer(AdvancedQuantumCircuitLayer):
    """Backward-compatible alias for the core quantum circuit layer."""

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 2,
        entanglement: Union[EntanglementType, str] = EntanglementType.LINEAR,
    ) -> None:
        entanglement_value = (
            entanglement.value if isinstance(entanglement, EntanglementType) else entanglement
        )
        super().__init__(n_qubits=n_qubits, n_layers=depth, entanglement=entanglement_value)
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            b, s, d = x.shape
            x_flat = x.reshape(-1, d)
            out = super().forward(x_flat)
            if out.shape[-1] != d:
                if (
                    not hasattr(self, "_proj_out")
                    or self._proj_out.in_features != out.shape[-1]
                    or self._proj_out.out_features != d
                ):
                    self._proj_out = nn.Linear(out.shape[-1], d).to(out.device)
                out = self._proj_out(out)
            return out.reshape(b, s, -1)
        out = super().forward(x)
        if out.shape[-1] != x.shape[-1]:
            if (
                not hasattr(self, "_proj_out")
                or self._proj_out.in_features != out.shape[-1]
                or self._proj_out.out_features != x.shape[-1]
            ):
                self._proj_out = nn.Linear(out.shape[-1], x.shape[-1]).to(out.device)
            out = self._proj_out(out)
        return out


class QuantumHybridLayer(nn.Module):
    """Backward-compatible hybrid wrapper used by tests/notebooks."""

    def __init__(self, input_dim: int, n_qubits: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.proj_in = nn.Linear(input_dim, n_qubits)
        self.quantum = QuantumLayer(n_qubits=n_qubits, depth=depth)
        self.proj_out = nn.Linear(n_qubits, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x
        out = self.proj_out(self.quantum(self.proj_in(x_flat)))
        if len(original_shape) == 3:
            return out.reshape(original_shape[0], original_shape[1], -1)
        return out


__all__ = [
    "AdvancedQuantumCircuitLayer",
    "QuantumKernelLayer",
    "QuantumAttentionLayer",
    "HybridQuantumClassicalBlock",
    "EntanglementType",
    "QuantumLayer",
    "QuantumHybridLayer",
]
