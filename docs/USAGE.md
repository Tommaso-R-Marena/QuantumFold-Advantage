# QuantumFold-Advantage Usage Guide

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended
- PennyLane-compatible quantum backend

### Setup

```bash
# Clone repository
git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
cd QuantumFold-Advantage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Optional: Quantum Hardware Access

```bash
# IBM Quantum
pip install qiskit qiskit-ibm-runtime

# AWS Braket
pip install amazon-braket-sdk

# Configure credentials (see respective documentation)
```

## Quick Start

### 1. Run Demo

```bash
python run_demo.py
```

This will:
- Load a sample protein
- Run QuantumFold prediction
- Display structure metrics
- Save results to `outputs/`

### 2. Predict Structure from Sequence

```python
from src.pipeline import QuantumFoldPipeline

# Initialize pipeline
pipeline = QuantumFoldPipeline(
    checkpoint='checkpoints/quantumfold_best.pt',
    use_quantum=True
)

# Predict structure
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQT LPGFGDSIEAQCGTSVNVHSSLRDILNQITKPNDVYSFSLASNSD FLDSKISNLTDENIHPFEVAFRIQDVDAVQKALVPLLEKKEVVGSRIY"
results = pipeline.predict(sequence)

print(f"Predicted coordinates shape: {results['coordinates'].shape}")
print(f"Confidence score: {results['confidence']:.3f}")
```

### 3. Predict from PDB File

```python
from src.data import load_protein_structure
from src.model import QuantumFoldModel
import torch

# Load model
model = QuantumFoldModel(use_quantum=True)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()

# Load structure
features = load_protein_structure('data/example.pdb')

# Predict
with torch.no_grad():
    outputs = model(features)

coordinates = outputs['coordinates'].numpy()
```

## Training

### Prepare Training Data

1. **Download PDB structures**:
```bash
python scripts/download_pdb.py --list data/training_list.txt --output data/pdb/
```

2. **Preprocess data**:
```python
from src.data import ProteinDataset

dataset = ProteinDataset(
    data_dir='data/pdb/',
    max_seq_len=512,
    augment=True
)
```

### Train Model

```bash
python src/train.py \
    --data_dir data/train/ \
    --val_dir data/validation/ \
    --output_dir checkpoints/ \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --use_quantum \
    --n_qubits 4
```

### Training Configuration

```python
training_config = {
    # Model architecture
    'n_layers': 6,
    'embed_dim': 128,
    'n_heads': 8,
    'n_qubits': 4,
    'use_quantum': True,
    
    # Training parameters
    'batch_size': 4,
    'learning_rate': 1e-4,
    'epochs': 50,
    'warmup_steps': 1000,
    
    # Loss weights
    'lambda_coord': 1.0,
    'lambda_dist': 0.5,
    'lambda_angle': 0.3,
    
    # Optimization
    'optimizer': 'adam',
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    
    # Scheduler
    'scheduler': 'cosine',
    'min_lr': 1e-6,
}
```

## Evaluation

### Evaluate on Validation Set

```bash
python scripts/evaluate_model.py \
    --checkpoint checkpoints/epoch_50.pt \
    --data_dir data/validation/ \
    --output evaluation_results.json
```

### Run Comprehensive Benchmark

```bash
python scripts/run_benchmark.py \
    --test_set data/casp15_test.json \
    --model_checkpoint checkpoints/best_model.pt \
    --compare_alphafold \
    --alphafold_dir data/alphafold_predictions/ \
    --output_dir results/benchmarks/
```

### Custom Evaluation

```python
from src.benchmarks import ProteinStructureEvaluator
import numpy as np

# Load predictions and ground truth
coords_pred = np.load('predictions/protein_pred.npy')
coords_true = np.load('data/protein_true.npy')

# Evaluate
evaluator = ProteinStructureEvaluator()
metrics = evaluator.evaluate_structure(
    coords_pred,
    coords_true,
    sequence_length=len(coords_true)
)

print(f"TM-score: {metrics.tm_score:.3f}")
print(f"RMSD: {metrics.rmsd:.2f} Å")
print(f"GDT_TS: {metrics.gdt_ts:.1f}")
print(f"lDDT: {metrics.lddt:.3f}")
```

## Advanced Usage

### Custom Quantum Circuits

```python
from src.quantum_layers import QuantumCircuitLayer
import pennylane as qml

# Define custom circuit
class CustomQuantumLayer(QuantumCircuitLayer):
    def _circuit(self, inputs, weights):
        # Custom encoding
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(inputs[i], wires=i)
        
        # Custom entangling
        for i in range(self.n_qubits):
            qml.CZ(wires=[i, (i+1) % self.n_qubits])
        
        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

# Use in model
quantum_layer = CustomQuantumLayer(n_qubits=6, n_layers=3)
```

### Ablation Studies

```python
# Train without quantum layers
model_classical = QuantumFoldModel(use_quantum=False)

# Train with quantum layers
model_quantum = QuantumFoldModel(use_quantum=True, n_qubits=4)

# Compare performance
results_classical = evaluate(model_classical, test_data)
results_quantum = evaluate(model_quantum, test_data)

print(f"Classical TM-score: {results_classical['tm_score']:.3f}")
print(f"Quantum TM-score: {results_quantum['tm_score']:.3f}")
print(f"Improvement: {results_quantum['tm_score'] - results_classical['tm_score']:.3f}")
```

### Batch Prediction

```python
from src.pipeline import QuantumFoldPipeline
from pathlib import Path

pipeline = QuantumFoldPipeline(checkpoint='checkpoints/best_model.pt')

# Process multiple sequences
sequences = {
    'protein1': 'MKTAYIAKQR...',
    'protein2': 'MNIFEMLRID...',
    'protein3': 'MGSSHHHHH...',
}

for name, seq in sequences.items():
    results = pipeline.predict(seq)
    pipeline.save_structure(
        results['coordinates'],
        output_path=f'outputs/{name}.pdb'
    )
    print(f"{name}: Confidence = {results['confidence']:.3f}")
```

## Visualization

### Plot Predicted Structure

```python
from src.visualize import plot_structure, plot_distance_map
import matplotlib.pyplot as plt

# Plot 3D structure
fig = plot_structure(coordinates, sequence)
plt.savefig('structure.png', dpi=300)

# Plot predicted distance map
fig = plot_distance_map(predicted_distances, true_distances)
plt.savefig('distances.png', dpi=300)
```

### PyMOL Visualization

```python
# Save structure
pipeline.save_structure(
    coordinates,
    output_path='predicted_structure.pdb'
)

# Load in PyMOL
# pymol predicted_structure.pdb
```

## Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Reduce batch size
python src/train.py --batch_size 2

# Use gradient accumulation
python src/train.py --accumulation_steps 4
```

**Quantum Simulation Slow**
```python
# Use lightning.qubit for faster simulation
from src.quantum_layers import QuantumCircuitLayer

layer = QuantumCircuitLayer(
    n_qubits=4,
    device_name='lightning.qubit'  # C++ backend
)
```

**CUDA Errors**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
export CUDA_VISIBLE_DEVICES=""
python src/train.py --device cpu
```

## Performance Tips

1. **Use mixed precision training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
```

2. **Optimize data loading**:
```python
dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,  # Parallel loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2
)
```

3. **Profile execution**:
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Citation

If you use QuantumFold-Advantage in your research, please cite:

```bibtex
@software{quantumfold2026,
  author = {Marena, Tommaso R.},
  title = {QuantumFold-Advantage: Hybrid Quantum-Classical Protein Folding},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Tommaso-R-Marena/QuantumFold-Advantage}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/issues)
- **Documentation**: [Full Docs](docs/)
- **Examples**: See `examples/` directory
