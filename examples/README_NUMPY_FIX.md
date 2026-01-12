# NumPy Compatibility Fix

## Problem

Google Colab pre-installs NumPy 2.0+, but PennyLane's dependency `autograd` requires NumPy <2.0 due to binary ABI changes. This causes:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

## Solution

All example notebooks now include a **comprehensive fix** that:

1. **Uninstalls conflicting packages** that force NumPy 2.0:
   - JAX/JAXlib
   - SHAP
   - PyTensor
   - OpenCV (will reinstall compatible version)

2. **Installs NumPy <2.0 with `--no-deps`** to prevent auto-upgrades

3. **Forces NumPy version** after all other package installs

4. **Auto-detects** if NumPy 2.x slips through and provides fix

5. **Requires runtime restart** after installation (critical!)

## Usage

### In Notebooks

1. Run the installation cell (Cell 3)
2. **RESTART RUNTIME** (Runtime > Restart runtime)
3. Run the verification cell (Cell 4)
4. If verification fails, it will auto-fix and tell you to restart again

### If You Still Get Errors

Run this in a cell:

```python
# Emergency fix
!pip uninstall -y jax jaxlib shap pytensor
!pip install --force-reinstall --no-cache-dir --no-deps 'numpy>=1.23.0,<2.0.0'
!pip install --force-reinstall --no-cache-dir 'autograd>=1.6.2'
!pip install --force-reinstall --no-cache-dir --no-deps 'pennylane>=0.32.0'

# RESTART RUNTIME NOW!
import os
os.kill(os.getpid(), 9)  # Force restart
```

## Why This Happens

NumPy 2.0 changed the C ABI (Application Binary Interface) for `numpy.dtype`. Packages compiled against NumPy <2.0 (like `autograd 1.6.2`) cannot import the new NumPy 2.x without recompilation.

## Fixed Notebooks

- ✅ `complete_benchmark.ipynb`
- ✅ `colab_quickstart.ipynb`
- ✅ `01_getting_started.ipynb`
- ✅ `02_quantum_vs_classical.ipynb`
- ✅ `03_advanced_visualization.ipynb`

## Local Installation

Local users should install with:

```bash
pip install 'numpy>=1.23.0,<2.0.0' 'autograd>=1.6.2' 'pennylane>=0.32.0'
```

## Future

Once `autograd` releases a version compatible with NumPy 2.0, this fix will no longer be needed.
