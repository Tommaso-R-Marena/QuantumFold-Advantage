# Universal Installation Cell Template for Google Colab

## Copy-Paste This Into ANY Notebook!

Replace your existing installation cell with this code:

```python
if IN_COLAB:
    !git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
    %cd QuantumFold-Advantage
    !pip install -q -e '.[protein-lm]'
    !pip install -q py3Dmol nglview biopython imageio
    print('✅ Installation complete!')
    print('⚠️  Restarting runtime to apply numpy 2.0 upgrade...')
    print('    After restart, skip this cell and continue from imports.')
    import os
    import time
    time.sleep(2)
    os.kill(os.getpid(), 9)
```

## What This Does

1. **Clones the repository** with updated `requirements.txt` (numpy>=2.0)
2. **Installs all dependencies** including visualization packages
3. **Automatically restarts the runtime** to load numpy 2.0
4. **Prevents binary incompatibility errors** with pandas, seaborn, etc.

## Usage Instructions

### First Run
1. Run the installation cell
2. Runtime will automatically restart (takes ~5 seconds)
3. **DO NOT re-run installation cell**
4. Continue from the imports cell

### Subsequent Runs
- Skip the installation cell entirely
- Start from imports cell

## Why Runtime Restart is Needed

Google Colab pre-loads numpy 1.x in memory. When you `pip install` numpy 2.0, Python's import system still finds the cached old version first. The restart clears memory and loads the new version.

## Applies To All Notebooks

This template works for:
- `01_getting_started.ipynb`
- `02_quantum_advantage_benchmark.ipynb`
- `03_atomic_visualization_showcase.ipynb`
- `02_a100_ULTIMATE_MAXIMIZED.ipynb`
- `colab_quickstart.ipynb`
- All other example notebooks

## Already Fixed

The following have been updated in the repository:
✅ `requirements.txt` - Now specifies `numpy>=2.0.0`
✅ All documentation updated
✅ Automatic fix script available at `examples/fix_numpy_colab.py`

---

**Last Updated:** February 4, 2026  
**Issue:** Binary incompatibility between numpy 1.x and 2.x  
**Solution:** Auto-restart after pip install
