# QuantumFold-Advantage Architecture

## Overview

QuantumFold-Advantage implements a hybrid quantum-classical neural network architecture for protein structure prediction. The model combines classical deep learning components with variational quantum circuits to potentially capture quantum correlations in protein conformations.

## Model Architecture

### High-Level Design

```
Input → Feature Extraction → Hybrid Encoder → Structure Module → 3D Coordinates
         ↓                      ↓                  ↓
      Sequence            Quantum Layers    Classical Refinement
      MSA                Classical Layers   Distance Prediction
      Templates          Attention          Angle Prediction
```

### Components

#### 1. Feature Extraction Layer

**Purpose**: Convert raw protein data into model-compatible representations.

**Inputs**:
- Amino acid sequence (one-hot encoded)
- Multiple sequence alignment (MSA)
- Template structures (optional)
- Evolutionary features

**Outputs**:
- Residue embeddings (dimension: 128-256)
- Pairwise features (distance predictions, contacts)
- MSA representations

**Implementation**: `src/data_processing.py`

#### 2. Quantum Circuit Layers

**Purpose**: Process features through parameterized quantum circuits to potentially capture non-classical correlations.

**Architecture**:
```python
QuantumCircuitLayer(
    n_qubits=4,          # Number of qubits
    n_layers=2-3,        # Variational layers
    rotation_gates=[RY, RZ],  # Rotation gates
    entangling=CNOT_ladder    # Entangling structure
)
```

**Key Features**:
- **Input Encoding**: Classical features → Quantum states via rotation gates
- **Variational Layers**: Trainable rotation gates + entangling operations
- **Measurement**: Expectation values of Pauli-Z operators
- **Integration**: PyTorch-compatible via PennyLane

**Implementation**: `src/quantum_layers.py::QuantumCircuitLayer`

#### 3. Hybrid Quantum-Classical Blocks

**Purpose**: Parallel processing through quantum and classical branches with learned fusion.

**Architecture**:
```
                     Input Features
                     /            \
            Classical Branch    Quantum Branch
                 |                    |
            Linear + ReLU      Quantum Circuit
                 |                    |
            LayerNorm          Linear Projection
                  \                  /
                   \                /
                    Fusion Layer
                         |
                  Residual Connection
                         |
                      Output
```

**Implementation**: `src/quantum_layers.py::HybridQuantumClassicalBlock`

#### 4. Quantum-Enhanced Attention

**Purpose**: Use quantum circuits to modulate attention weights for capturing long-range dependencies.

**Mechanism**:
1. Classical attention computation (Q, K, V projections)
2. Quantum enhancement of attention weights
3. Apply modulated attention to values

**Implementation**: `src/quantum_layers.py::QuantumAttentionLayer`

#### 5. Structure Prediction Module

**Purpose**: Convert learned representations to 3D coordinates.

**Components**:
- **Distance Predictor**: Pairwise distance maps
- **Angle Predictor**: Backbone dihedral angles (φ, ψ, ω)
- **Coordinate Builder**: Construct 3D structure from distances/angles

**Implementation**: `src/model.py`

## Quantum Computing Backend

### Device Options

1. **Simulation** (Default for Development)
   - `default.qubit`: Standard simulator
   - `lightning.qubit`: High-performance C++ simulator
   - Suitable for: 4-8 qubits

2. **Hardware** (Future Deployment)
   - IBM Quantum
   - AWS Braket
   - IonQ
   - Requires device-specific configuration

### Quantum Circuit Design

**Encoding Strategy**: Angle encoding
```
for i in range(n_qubits):
    RY(classical_input[i], qubit=i)
```

**Variational Ansatz**: Hardware-efficient ansatz
```
for layer in range(n_layers):
    # Rotation layer
    for qubit in range(n_qubits):
        RY(θ[layer, qubit, 0], qubit)
        RZ(θ[layer, qubit, 1], qubit)
    
    # Entangling layer
    for qubit in range(n_qubits - 1):
        CNOT(qubit, qubit+1)
```

**Parameter Count**: `n_qubits × n_rotations × n_layers`

## Training Strategy

### Loss Functions

1. **Coordinate Loss** (Primary)
   ```
   L_coord = MSE(coords_pred, coords_true)
   ```

2. **Distance Loss**
   ```
   L_dist = MSE(distances_pred, distances_true)
   ```

3. **Angle Loss**
   ```
   L_angle = MSE(angles_pred, angles_true)
   ```

4. **Combined Loss**
   ```
   L_total = λ₁·L_coord + λ₂·L_dist + λ₃·L_angle
   ```

### Optimization

- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 4-8 (limited by quantum simulation)
- **Gradient Clipping**: max_norm=1.0

### Hyperparameters

```python
model_config = {
    'n_layers': 6,              # Transformer layers
    'embed_dim': 128,           # Embedding dimension
    'n_heads': 8,               # Attention heads
    'n_qubits': 4,              # Qubits per quantum layer
    'quantum_layers': 2,        # Variational depth
    'dropout': 0.1,             # Dropout rate
    'use_quantum': True,        # Enable quantum layers
}
```

## Benchmarking Metrics

### Primary Metrics

1. **TM-score** (Template Modeling score)
   - Range: [0, 1]
   - >0.5: Similar fold
   - >0.6: Same topology

2. **RMSD** (Root Mean Square Deviation)
   - Unit: Ångströms (Å)
   - Lower is better
   - Typical: 2-6 Å for good predictions

3. **GDT_TS** (Global Distance Test - Total Score)
   - Range: [0, 100]
   - Percentage of residues within distance thresholds
   - Industry standard for CASP

4. **lDDT** (Local Distance Difference Test)
   - Range: [0, 1]
   - Measures local structure quality
   - Less sensitive to domain orientation

### Implementation

See `src/benchmarks.py` for complete metric implementations.

## Comparison with AlphaFold-3

### Similarities

- Transformer-based architecture
- MSA processing
- Structure module with distance/angle prediction

### Key Differences

| Aspect | AlphaFold-3 | QuantumFold |
|--------|-------------|-------------|
| Core Architecture | Evoformer | Hybrid Quantum-Classical |
| Parameter Updates | Classical gradient descent | Quantum-aware optimization |
| Attention Mechanism | Standard multi-head | Quantum-enhanced |
| Feature Processing | Fully classical | Parallel quantum/classical |
| Theoretical Basis | Deep learning | Quantum computing + DL |

### Potential Advantages

1. **Quantum Correlations**: May capture entangled conformational states
2. **Compact Representations**: Quantum states offer exponential capacity
3. **Novel Search**: Quantum layers explore different solution spaces

### Limitations

1. **Scalability**: Current quantum hardware limited to 4-8 qubits
2. **Noise**: NISQ devices introduce errors
3. **Speed**: Quantum simulation slower than classical for small problems

## Future Directions

### Near-Term (6-12 months)

1. **Expand Quantum Integration**
   - Increase qubits to 8-12
   - Test on real quantum hardware
   - Quantum error mitigation

2. **Enhanced Features**
   - Template-based modeling
   - Confidence prediction
   - Multi-chain complexes

3. **Benchmarking**
   - CASP evaluation
   - CAMEO continuous assessment
   - Comparison with latest AlphaFold variants

### Long-Term (1-2 years)

1. **Quantum Advantage Demonstration**
   - Identify specific protein classes where quantum helps
   - Rigorous statistical testing
   - Publication-ready results

2. **Production Deployment**
   - Optimized inference pipeline
   - Web server for predictions
   - API access

3. **Novel Applications**
   - Protein design
   - Drug discovery
   - Antibody modeling

## References

1. Jumper et al., "Highly accurate protein structure prediction with AlphaFold," Nature 2021
2. Abramson et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3," Nature 2024
3. Biamonte et al., "Quantum machine learning," Nature 2017
4. Cerezo et al., "Variational quantum algorithms," Nature Reviews Physics 2021
5. Senior et al., "Improved protein structure prediction using potentials from deep learning," Nature 2020
