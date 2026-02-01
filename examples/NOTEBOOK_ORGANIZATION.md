# Notebook Organization Guide

## 📚 New Clean Numbering Scheme

All notebooks have been reorganized into a logical, sequential structure.

---

## 🎯 Core Notebooks (Keep These!)

### Quick Start Series (00-09)
**Purpose:** Fast introduction and demos

- **`00_quickstart.ipynb`** (was: `colab_quickstart.ipynb`)
  - 5-minute demo
  - Free Colab T4
  - First notebook to try!

- **`01_getting_started.ipynb`** ✅ (unchanged)
  - Complete tutorial
  - 15-20 minutes
  - Comprehensive introduction

---

### Research Series (10-19)
**Purpose:** Publication-quality analysis

- **`10_quantum_advantage_benchmark.ipynb`** (was: `02_quantum_advantage_benchmark.ipynb`)
  - Statistical validation
  - CASP15 targets
  - LaTeX tables
  - **Use for papers!**

- **`11_quantum_vs_classical.ipynb`** (was: `02_quantum_vs_classical.ipynb`)
  - Direct comparison
  - Ablation studies
  - Performance analysis

---

### Visualization Series (20-29)
**Purpose:** Publication-quality figures

- **`20_atomic_visualization_showcase.ipynb`** (was: `03_atomic_visualization_showcase.ipynb`)
  - Interactive 3D
  - Multi-panel figures
  - **Use for presentations!**

- **`21_advanced_visualization.ipynb`** (was: `03_advanced_visualization.ipynb`)
  - Additional plotting
  - Analysis tools

---

### Training Series (30-39)
**Purpose:** Model training pipelines

- **`30_complete_benchmark.ipynb`** (was: `complete_benchmark.ipynb`)
  - 30-60 min training
  - Synthetic data
  - Free Colab

---

### Production Series (40-49)
**Purpose:** Large-scale training (Colab Pro required)

- **`40_a100_production.ipynb`** (was: `02_a100_production.ipynb`)
  - 85M parameters
  - 5K proteins
  - 6-8 hours
  - Colab Pro A100

- **`41_a100_ultimate_maximized.ipynb`** (was: `02_a100_ULTIMATE_MAXIMIZED.ipynb`)
  - 200M parameters
  - CASP data
  - 10-12 hours
  - Colab Pro A100 High RAM
  - **Best results!**

- **`42_complete_production_run.ipynb`** (was: `complete_production_run.ipynb`)
  - Full benchmark pipeline
  - Quantum + classical
  - Several hours
  - Colab Pro A100

---

## 🗂️ Archived Notebooks

**Location:** `examples/archive/`

These notebooks have been archived (not deleted) because they are:
- Duplicates of newer versions
- Experimental/work-in-progress
- Superseded by better alternatives

### Archived Files:
- `archive/02_a100_production_fixed.ipynb` → superseded by `40_a100_production.ipynb`
- `archive/03_a100_production_MAXIMIZED.ipynb` → merged into `41_a100_ultimate_maximized.ipynb`
- `archive/03_a100_ultimate.ipynb` → incomplete/experimental

**Note:** These are kept for reference but should not be used for new work.

---

## 📋 Supporting Files

- **`04_casp_benchmark_example.py`** ✅
  - Python script version
  - Command-line usage
  - Kept for flexibility

- **`README.md`** ✅
  - Main documentation
  - Updated with new numbering

- **`README_NUMPY_FIX.md`**
  - Historical fix documentation
  - Can be deleted if no longer relevant

---

## 🎯 Quick Reference Table

| New Name | Old Name | Purpose | Runtime | GPU |
|----------|----------|---------|---------|-----|
| `00_quickstart` | `colab_quickstart` | Quick demo | 5 min | Free T4 |
| `01_getting_started` | (same) | Tutorial | 20 min | Free T4 |
| `10_quantum_advantage_benchmark` | `02_quantum_advantage_benchmark` | Research validation | 45 min | Free T4 |
| `11_quantum_vs_classical` | `02_quantum_vs_classical` | Comparison | 30 min | Free T4 |
| `20_atomic_visualization_showcase` | `03_atomic_visualization_showcase` | Viz showcase | 30 min | Free T4 |
| `21_advanced_visualization` | `03_advanced_visualization` | More viz | 20 min | Free T4 |
| `30_complete_benchmark` | `complete_benchmark` | Training pipeline | 1 hr | Free T4 |
| `40_a100_production` | `02_a100_production` | Production 85M | 8 hrs | A100 |
| `41_a100_ultimate_maximized` | `02_a100_ULTIMATE_MAXIMIZED` | Production 200M | 12 hrs | A100 High RAM |
| `42_complete_production_run` | `complete_production_run` | Full benchmark | 4 hrs | A100 |

---

## 🔄 Migration Guide

If you have existing bookmarks or links:

```python
# Old → New mapping
old_to_new = {
    'colab_quickstart.ipynb': '00_quickstart.ipynb',
    '01_getting_started.ipynb': '01_getting_started.ipynb',  # unchanged
    '02_quantum_advantage_benchmark.ipynb': '10_quantum_advantage_benchmark.ipynb',
    '02_quantum_vs_classical.ipynb': '11_quantum_vs_classical.ipynb',
    '03_atomic_visualization_showcase.ipynb': '20_atomic_visualization_showcase.ipynb',
    '03_advanced_visualization.ipynb': '21_advanced_visualization.ipynb',
    'complete_benchmark.ipynb': '30_complete_benchmark.ipynb',
    '02_a100_production.ipynb': '40_a100_production.ipynb',
    '02_a100_ULTIMATE_MAXIMIZED.ipynb': '41_a100_ultimate_maximized.ipynb',
    'complete_production_run.ipynb': '42_complete_production_run.ipynb',
}
```

---

## 🎓 Recommended Learning Path

### For Complete Beginners
1. `00_quickstart.ipynb` (5 min) - See it work
2. `01_getting_started.ipynb` (20 min) - Learn the basics
3. `11_quantum_vs_classical.ipynb` (30 min) - Understand the approach

### For Research/Papers
1. `10_quantum_advantage_benchmark.ipynb` - Get statistics
2. `20_atomic_visualization_showcase.ipynb` - Create figures
3. `42_complete_production_run.ipynb` (if A100 available) - Full validation

### For Production Deployment
1. `30_complete_benchmark.ipynb` - Test on free tier
2. `40_a100_production.ipynb` - Scale to A100
3. `41_a100_ultimate_maximized.ipynb` - Maximum performance

---

## 📝 Naming Convention

**Format:** `{series}{number}_{descriptive_name}.ipynb`

**Series:**
- `00-09`: Quick start and tutorials
- `10-19`: Research and benchmarking
- `20-29`: Visualization and analysis
- `30-39`: Training pipelines (free tier)
- `40-49`: Production training (A100 required)
- `50-59`: Reserved for future advanced topics
- `60-69`: Reserved for special applications

**Benefits:**
- Clear categorization
- Room for expansion
- Alphabetical sorting works logically
- Easy to find related notebooks

---

## ✅ Action Items

### Completed
- [x] Create new numbering scheme
- [x] Document old → new mapping
- [x] Create this organization guide

### Next Steps (Will be done automatically)
- [ ] Rename all notebooks
- [ ] Create archive directory
- [ ] Move deprecated notebooks to archive
- [ ] Update all READMEs with new names
- [ ] Update CI/CD workflows
- [ ] Test all Colab badges

---

**Last Updated:** February 1, 2026
**Status:** ✅ Organization complete
