# 🔧 Numpy Compatibility Fix for Google Colab

## Problem

Google Colab's pre-installed packages (JAX, OpenCV, PyTensor, etc.) now require **numpy>=2.0**, but older sessions may have numpy 1.x cached in memory, causing binary incompatibility errors:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

## Solution

As of **February 2026**, all QuantumFold-Advantage notebooks use `numpy>=2.0.0` in requirements.txt.

### For Users Running Notebooks

**Option 1: Automatic Fix (Recommended)**

Add this cell RIGHT AFTER the installation cell:

```python
# Auto-restart runtime to load updated numpy
import os
print("⚠️  Restarting runtime to apply numpy upgrade...")
os.kill(os.getpid(), 9)
```

After restart:
1. **DO NOT** re-run the installation cell
2. Start from the imports cell
3. Everything will work!

**Option 2: Manual Restart**

1. Run installation cell
2. **Runtime → Restart runtime** (from menu)
3. Skip installation cell
4. Run imports cell

### For Notebook Developers

To prevent this issue in new notebooks, use this standardized installation cell:

```python
# Installation Cell (Run once, then restart runtime)
if IN_COLAB:
    !git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
    %cd QuantumFold-Advantage
    !pip install -q -e '.[protein-lm]'
    
    # Install visualization dependencies
    !pip install -q py3Dmol nglview biopython imageio
    
    print('✅ Installation complete!')
    print('⚠️  IMPORTANT: Runtime will restart automatically...')
    print('   After restart, skip this cell and run the next one.')
    
    # Auto-restart
    import os
    import time
    time.sleep(2)
    os.kill(os.getpid(), 9)
```

## Why This Happens

1. **Colab preloads numpy 1.x** in the runtime environment
2. **Your package installs numpy 2.0** as specified in requirements.txt
3. **Python's import system** finds the old cached numpy first
4. **Binary incompatibility** between compiled C extensions (pandas, seaborn)

## Fixed Notebooks

All notebooks in `/examples/` now use `numpy>=2.0.0`:

- ✅ `01_getting_started.ipynb`
- ✅ `02_quantum_advantage_benchmark.ipynb`  
- ✅ `03_atomic_visualization_showcase.ipynb`
- ✅ `02_a100_ULTIMATE_MAXIMIZED.ipynb`
- ✅ `colab_quickstart.ipynb`
- ✅ All other notebooks

## Testing

After applying the fix, verify numpy version:

```python
import numpy as np
print(f"Numpy version: {np.__version__}")
assert np.__version__.startswith('2.'), "Numpy 2.0+ required!"
```

Should output:
```
Numpy version: 2.0.0 (or higher)
```

## Still Having Issues?

1. **Clear Colab cache**: Runtime → Disconnect and delete runtime
2. **Start fresh session**: Reconnect and run from beginning
3. **Report issue**: [Open GitHub issue](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/issues) with full error trace

---

**Last Updated:** February 3, 2026  
**Affects:** Google Colab environments only  
**Fix Status:** ✅ Resolved in all current notebooks
