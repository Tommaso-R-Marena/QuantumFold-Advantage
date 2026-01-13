# Google Colab Setup Guide

## ✅ Solution: NumPy 2.x + PennyLane 0.38+

As of January 2026, Google Colab ships with **NumPy 2.x** as a core dependency for many packages. This guide explains how to run QuantumFold-Advantage notebooks in this environment.

---

## Quick Start

### Option 1: Use Example Notebooks (Recommended)

All example notebooks are pre-configured with the correct installation:

1. **[01_getting_started.ipynb](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/01_getting_started.ipynb)** - Complete tutorial
2. **[colab_quickstart.ipynb](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/colab_quickstart.ipynb)** - Fast intro
3. **[complete_benchmark.ipynb](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/complete_benchmark.ipynb)** - Full pipeline

Simply click "Open in Colab" and run all cells!

### Option 2: Manual Setup

```python
# In Google Colab
!git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
%cd QuantumFold-Advantage

# Install dependencies (NumPy 2.x compatible)
!pip install --quiet 'pennylane>=0.38' 'autoray>=0.6.11'
!pip install --quiet matplotlib 'seaborn>=0.13' plotly
!pip install --quiet statsmodels biopython requests tqdm

# Optional: ESM-2 for protein embeddings
!pip install --quiet fair-esm transformers
```

---

## Why NumPy 2.x?

### Colab's NumPy 2.x Dependencies

Google Colab now requires NumPy 2.x for these core packages:

- **opencv-python** (4.12.0.88): `numpy>=2, <2.3.0`
- **jax** (0.7.2): `numpy>=2.0`
- **jaxlib** (0.7.2): `numpy>=2.0`
- **pytensor** (2.36.3): `numpy>=2.0`
- **rasterio** (1.5.0): `numpy>=2.0`
- **shap** (0.50.0): `numpy>=2`

Downgrading NumPy to 1.x creates dependency conflicts with these packages.

### PennyLane Compatibility

**PennyLane 0.38+** (released October 2024) added full NumPy 2.x support:

```
PennyLane 0.38.0 Release Notes:
- NumPy 2.0 compatibility
- Updated core operations for new NumPy API
- Binary compatibility with NumPy 2.x
```

This makes NumPy 2.x + PennyLane 0.38+ the **optimal** solution.

---

## Version Requirements

### Minimum Versions

| Package | Minimum Version | Why |
|---------|----------------|-----|
| NumPy | 2.0.0 | Colab requirement |
| PennyLane | 0.38.0 | NumPy 2.x support |
| PyTorch | 2.0+ | Automatic in Colab |
| Seaborn | 0.13.0 | NumPy 2.x compatibility |

### Tested Versions (January 2026)

```python
import numpy as np
import pennylane as qml
import torch

print(f"NumPy: {np.__version__}")        # 2.4.1
print(f"PennyLane: {qml.__version__}")  # 0.39.0
print(f"PyTorch: {torch.__version__}")  # 2.5.1+cu121
```

---

## Testing

### Automated Test Suite

Run the comprehensive test notebook:

**[Test Suite](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/tests/test_notebooks_colab.ipynb)**

This verifies:
- ✅ Installation works
- ✅ All imports succeed
- ✅ Quantum circuits execute
- ✅ Models train
- ✅ Visualizations generate

### Manual Verification

```python
# Verify NumPy 2.x
import numpy as np
assert int(np.__version__.split('.')[0]) >= 2, "NumPy 2.x required"
print(f"✅ NumPy {np.__version__}")

# Verify PennyLane
import pennylane as qml
version = tuple(map(int, qml.__version__.split('.')[:2]))
assert version >= (0, 38), "PennyLane 0.38+ required"
print(f"✅ PennyLane {qml.__version__}")

# Test quantum circuit
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(x):
    qml.RY(x, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

result = circuit(0.5)
print(f"✅ Quantum circuit: {result:.3f}")
```

---

## Troubleshooting

### Error: "numpy.dtype size changed"

**Cause:** Mixing NumPy 1.x and 2.x binary extensions.

**Solution:** This should NOT occur with the new setup. If it does:

```python
# Force reinstall all NumPy-dependent packages
!pip uninstall -y numpy scipy pandas
!pip install --no-cache-dir numpy scipy pandas
```

### Error: "PennyLane does not support NumPy 2.x"

**Cause:** Old PennyLane version (<0.38).

**Solution:**

```python
!pip install --upgrade 'pennylane>=0.38'
```

### Error: "Module 'pennylane' has no attribute..."

**Cause:** API changes between PennyLane versions.

**Solution:** Update to latest stable:

```python
!pip install --upgrade pennylane
```

### Kernel Crash During Installation

**Cause:** Out of memory during package compilation.

**Solution:** Install packages sequentially:

```python
!pip install --quiet pennylane
!pip install --quiet autoray
!pip install --quiet matplotlib seaborn
# Continue with smaller batches
```

### Import Errors After Successful Installation

**Cause:** Python runtime has stale imports.

**Solution:** Restart runtime:

```python
import os
os.kill(os.getpid(), 9)  # Force restart
# Then re-run all cells
```

---

## Performance Optimization

### Enable GPU

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cpu':
    print("⚠️  Enable GPU: Runtime > Change runtime type > T4 GPU")
```

### Memory Management

```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Monitor GPU memory
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Batch Size Recommendations

| GPU | Max Batch Size | Sequence Length |
|-----|---------------|----------------|
| T4 (15GB) | 8 | 512 |
| A100 (40GB) | 32 | 1024 |
| V100 (16GB) | 16 | 512 |

---

## Advanced Configuration

### Custom PennyLane Device

```python
import pennylane as qml

# Use lightning.qubit for faster simulations
dev = qml.device('lightning.qubit', wires=10)

# For GPU acceleration (if available)
try:
    dev = qml.device('lightning.gpu', wires=10)
    print("✅ Using GPU-accelerated quantum simulator")
except:
    dev = qml.device('default.qubit', wires=10)
    print("⚠️  Using CPU quantum simulator")
```

### Mixed Precision Training

```python
import torch

# Enable automatic mixed precision
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## FAQs

### Q: Why not use NumPy 1.x?

**A:** Google Colab's core packages (opencv, jax, etc.) require NumPy 2.x. Downgrading causes widespread dependency conflicts.

### Q: Will this work locally?

**A:** Yes! The same setup works locally:

```bash
pip install 'pennylane>=0.38' 'numpy>=2.0' torch matplotlib seaborn
```

### Q: What about other quantum frameworks?

**A:** NumPy 2.x compatibility:

- ✅ **PennyLane 0.38+**: Full support
- ✅ **Qiskit 1.0+**: Full support
- ⚠️ **Cirq**: Partial support (check version)
- ❌ **PyQuil**: Not yet supported

### Q: Can I use ESM-2 embeddings?

**A:** Yes! Install with:

```python
!pip install fair-esm transformers
```

ESM-2 models are NumPy 2.x compatible.

---

## Citation

If you use QuantumFold-Advantage in your research:

```bibtex
@software{quantumfold2024,
  title={QuantumFold-Advantage: Quantum-Enhanced Protein Folding},
  author={Marena, Tommaso R.},
  year={2024},
  url={https://github.com/Tommaso-R-Marena/QuantumFold-Advantage}
}
```

---

## Support

- **Documentation:** [GitHub README](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage)
- **Issues:** [Report bugs](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/issues)
- **Discussions:** [Ask questions](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/discussions)

---

## Changelog

### January 2026

- ✅ Migrated to NumPy 2.x + PennyLane 0.38+
- ✅ Removed runtime restart requirement
- ✅ Added comprehensive test suite
- ✅ Updated all example notebooks

### Previous Approach (Deprecated)

~~NumPy 1.x with forced reinstallation~~ - No longer needed with PennyLane 0.38+

---

**Last Updated:** January 13, 2026  
**Tested On:** Google Colab (Python 3.12, NumPy 2.4.1, PennyLane 0.39.0)