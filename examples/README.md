# QuantumFold-Advantage Example Notebooks

This directory contains **research-grade Jupyter notebooks** demonstrating the full capabilities of QuantumFold-Advantage.

## 🚀 Quick Start Guide

### Best Notebooks to Start With

1. **For Quick Demo (5 min)** → `colab_quickstart.ipynb`
2. **For Rigorous Benchmarking (45 min)** → `02_quantum_advantage_benchmark.ipynb` ⭐
3. **For Beautiful Visualizations (30 min)** → `03_atomic_visualization_showcase.ipynb` ⭐
4. **For Production Training (8-10 hrs)** → `02_a100_ULTIMATE_MAXIMIZED.ipynb`

---

## 📚 Complete Notebook Catalog

### ⭐ Research-Grade Notebooks (NEW!)

#### `02_quantum_advantage_benchmark.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_quantum_advantage_benchmark.ipynb)

**Publication-quality statistical validation of quantum advantage**
- ✅ Real CASP15 targets
- ✅ Paired quantum vs. classical comparison  
- ✅ Wilcoxon signed-rank test
- ✅ Bootstrap confidence intervals (10K samples)
- ✅ Effect sizes (Cohen's d, rank-biserial)
- ✅ Power analysis
- ✅ LaTeX tables for papers
- ⏱️ **30-45 minutes** on free Colab (T4 GPU)

**Use for**: Research papers, grant proposals, rigorous evaluation

---

#### `03_atomic_visualization_showcase.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/03_atomic_visualization_showcase.ipynb)

**World-class protein structure visualization**
- 🧬 Interactive 3D molecular viewer (py3Dmol)
- 📊 Ramachandran plots with secondary structure
- 🗺️ Contact maps with annotations
- 🎯 Attention mechanism heatmaps
- ⚛️ Quantum circuit diagrams
- 🎬 Structure refinement animations
- 📈 Confidence visualization (pLDDT-style)
- ⏱️ **20-30 minutes** on free Colab

**Use for**: Publications, presentations, model analysis

---

### 🔥 Production Training Notebooks

#### `02_a100_ULTIMATE_MAXIMIZED.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_a100_ULTIMATE_MAXIMIZED.ipynb)

**200M parameter AlphaFold2-level pipeline**
- 🧬 CASP13/14/15 + RCSB + AlphaFoldDB data
- ⚡ Batch size 24, 100K steps
- 📦 167GB RAM - all embeddings in-memory
- 🎨 BF16 precision
- 🎯 Target: RMSD <1.5Å, TM-score >0.75
- ⏱️ **10-12 hours** on A100 High RAM
- 💾 **Requires**: Colab Pro with A100 (80GB GPU, 167GB RAM)

---

#### `02_a100_production.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_a100_production.ipynb)

**85M parameter production pipeline**
- 🧬 5000+ diverse PDB proteins
- 💪 Proper IPA architecture
- ⚡ Batch size 16, 50K steps  
- 🎯 Target: RMSD <2.0Å, TM-score >0.70
- ⏱️ **6-8 hours** on A100 High RAM
- 💾 **Requires**: Colab Pro with A100

---

#### `complete_production_run.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/complete_production_run.ipynb)

**Full quantum + classical benchmarking pipeline**
- Both quantum and classical training
- Statistical comparison
- Publication-ready analysis
- ⏱️ **Several hours** on A100

---

### 🎓 Learning & Quick Start

#### `colab_quickstart.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/colab_quickstart.ipynb)

**5-minute introduction**
- Quick model demo
- Basic structure prediction
- Simple visualization
- ⏱️ **5 minutes** on free Colab

---

#### `01_getting_started.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/01_getting_started.ipynb)

**Comprehensive tutorial**
- Model architecture overview
- Training workflow
- Evaluation metrics
- ⏱️ **15-20 minutes** on free Colab

---

#### `complete_benchmark.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/complete_benchmark.ipynb)

**30-60 minute training pipeline**
- Synthetic data training
- Basic benchmarking
- Metric calculation
- ⏱️ **30-60 minutes** on free Colab (T4 GPU)

---

### 🔧 Specialized Notebooks

#### `02_quantum_vs_classical.ipynb`
Direct comparison between quantum and classical approaches

#### `03_advanced_visualization.ipynb`  
Advanced plotting and analysis techniques

---

## 📊 Notebook Comparison

| Notebook | Runtime | GPU Required | Data | Output | Best For |
|----------|---------|--------------|------|--------|----------|
| `colab_quickstart` | 5 min | Free T4 | Synthetic | Demo | Quick intro |
| `02_quantum_advantage_benchmark` ⭐ | 45 min | Free T4 | CASP15 | Research paper | Publications |
| `03_atomic_visualization_showcase` ⭐ | 30 min | Free T4 | CASP15 | High-res figures | Presentations |
| `complete_benchmark` | 1 hr | Free T4 | Synthetic | Metrics | Learning |
| `02_a100_production` | 8 hrs | A100 | 5K proteins | Production model | Real training |
| `02_a100_ULTIMATE_MAXIMIZED` | 12 hrs | A100 High RAM | CASP+RCSB | SOTA model | Best results |

---

## 🎯 Use Case Guide

### I want to...

**...understand how it works** → Start with `colab_quickstart.ipynb`

**...write a research paper** → Use `02_quantum_advantage_benchmark.ipynb` for statistics

**...create beautiful figures** → Use `03_atomic_visualization_showcase.ipynb`

**...train a production model** → Use `02_a100_ULTIMATE_MAXIMIZED.ipynb` on Colab Pro

**...test quantum advantage** → Use `02_quantum_advantage_benchmark.ipynb`

**...learn the architecture** → Use `01_getting_started.ipynb`

---

## 💻 Hardware Requirements

### Free Colab (T4 GPU, 12GB RAM)
✅ `colab_quickstart.ipynb`
✅ `01_getting_started.ipynb`  
✅ `02_quantum_advantage_benchmark.ipynb` ⭐
✅ `03_atomic_visualization_showcase.ipynb` ⭐
✅ `complete_benchmark.ipynb`

### Colab Pro (A100 GPU, 40GB RAM)
✅ `02_a100_production.ipynb`
✅ `complete_production_run.ipynb`

### Colab Pro (A100 GPU, 167GB High RAM)
✅ `02_a100_ULTIMATE_MAXIMIZED.ipynb` (flagship)

---

## 📖 Documentation

Each notebook includes:
- 📝 Detailed markdown explanations
- 💡 Scientific methodology
- ⚙️ Hyperparameter descriptions  
- 📊 Expected results
- 🔍 Interpretation guidelines
- 💾 Output files and downloads

---

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)
Click any "Open in Colab" badge above. Everything installs automatically!

### Option 2: Local Jupyter
```bash
git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
cd QuantumFold-Advantage
pip install -e .[protein-lm]
jupyter notebook examples/
```

### Option 3: Docker
```bash
docker-compose up
# Access JupyterLab at http://localhost:8888
```

---

## 📚 Additional Resources

- **Main README**: [Project overview](../README.md)
- **API Documentation**: [src/README.md](../src/README.md)  
- **Training Guide**: [Advanced training](../docs/training.md)
- **Benchmarking Guide**: [Statistical validation](../docs/benchmarking.md)

---

## 🆘 Troubleshooting

### Common Issues

**"Out of memory" error**
- Reduce batch size
- Use gradient accumulation
- Try free Colab first

**"CASP data not found"**
- The loader will automatically generate synthetic targets
- Real CASP data requires manual download

**"ESM-2 download slow"**
- First run downloads ~3GB model
- Cached for future runs

**"py3Dmol not working"**
- May not render in some Jupyter environments
- Works best in Colab

---

## 🤝 Contributing

Found a bug or have a suggestion? Please [open an issue](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/issues)!

---

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details

---

## ⭐ New Features

### Recent Additions
- ✨ Research-grade statistical validation notebook
- ✨ World-class visualization showcase
- ✨ CASP data loader with real/synthetic targets
- ✨ Publication-quality figure generation
- ✨ LaTeX table export for papers
- ✨ Bootstrap confidence intervals
- ✨ Power analysis

Updated: February 2026
