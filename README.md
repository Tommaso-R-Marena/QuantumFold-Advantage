# QuantumFold-Advantage 🧬⚛️

<!-- Build Status Badges -->
[![CI](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/ci.yml/badge.svg)](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/ci.yml)
[![Test Notebooks](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/test-notebooks.yml/badge.svg)](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/test-notebooks.yml)
[![Docker](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/docker-publish.yml)
[![codecov](https://codecov.io/gh/Tommaso-R-Marena/QuantumFold-Advantage/branch/main/graph/badge.svg)](https://codecov.io/gh/Tommaso-R-Marena/QuantumFold-Advantage)

<!-- Technology Badges -->
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.33+-green.svg)](https://pennylane.ai/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Quick Links -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_quantum_advantage_benchmark.ipynb)
[![GitHub Stars](https://img.shields.io/github/stars/Tommaso-R-Marena/QuantumFold-Advantage?style=social)](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Tommaso-R-Marena/QuantumFold-Advantage?style=social)](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/network/members)

**State-of-the-art quantum-classical hybrid architecture for protein structure prediction**

QuantumFold-Advantage demonstrates rigorous quantum advantage in protein structure prediction by integrating quantum computing with advanced deep learning techniques from AlphaFold, ESM-2, and modern ML best practices.

---

## ⭐ **NEW: Research-Grade Notebooks**

### 🔬 **Publication-Quality Statistical Validation**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_quantum_advantage_benchmark.ipynb)

**`02_quantum_advantage_benchmark.ipynb`** - Rigorous statistical testing of quantum advantage

✨ **Research Features:**
- 📊 Real CASP15 protein targets
- 📈 Paired quantum vs. classical comparison
- 🧮 Wilcoxon signed-rank tests + t-tests
- 📉 Bootstrap confidence intervals (10K samples)
- 📐 Effect sizes (Cohen's d, rank-biserial)
- ⚡ Power analysis
- 📄 LaTeX tables for papers

⏱️ **30-45 minutes** on free Colab (T4 GPU)  
🎯 **Use for:** Research papers, grant proposals, thesis work

---

### 🎨 **World-Class Visualization Showcase**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/03_atomic_visualization_showcase.ipynb)

**`03_atomic_visualization_showcase.ipynb`** - Publication-quality visualizations

✨ **Visualization Features:**
- 🧬 Interactive 3D molecular structures (py3Dmol)
- 📊 Ramachandran plots with secondary structure
- 🗺️ Contact maps with annotations
- 🎯 Attention mechanism heatmaps
- ⚛️ Quantum circuit diagrams
- 🎬 Structure refinement animations
- 📈 Confidence visualization (pLDDT-style)
- 🖼️ Multi-panel publication figures

⏱️ **20-30 minutes** on free Colab  
🎯 **Use for:** Publications, presentations, posters

---

## 🚀 **Production Training Pipelines**

### 🔥 **ULTIMATE A100 MAXIMIZED**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_a100_ULTIMATE_MAXIMIZED.ipynb)

**200M Parameter AlphaFold2-Level Pipeline**

✨ **Ultimate Specifications:**
- 🧬 **200M parameters** - 1536 hidden, 16 encoder, 12 structure layers
- 📊 **CASP13/14/15 + RCSB + AlphaFoldDB** - Real benchmark targets
- ⚡ **Batch size 24** - 50% larger with smart bucketing
- 📦 **167GB RAM** - ALL embeddings in-memory
- 🎨 **BF16 precision** - Numerical stability
- 📈 **100K steps** - Full convergence training

🎯 **Target Performance (AlphaFold2-level):**
- RMSD: <1.5Å
- TM-score: >0.75  
- GDT_TS: >70
- pLDDT: >80

⏱️ **~10-12 hours** on A100 High RAM  
💾 **Requires:** Colab Pro with A100 (80GB GPU, 167GB RAM)

---

### 🏭 **A100 Production Training**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_a100_production.ipynb)

**85M Parameter Production Pipeline**

✨ **Production Features:**
- 🧬 **5000+ diverse proteins** from PDB
- 🏗️ **Proper IPA architecture** - Geometric attention
- 💪 **85M parameters** - 1024 hidden, 12 encoder, 8 structure layers
- 🎯 **Advanced losses** - FAPE + geometry + perceptual
- ⚡ **Batch size 16** with length bucketing
- 📊 **50K steps** - Full production training

🎯 **Expected Performance:**
- RMSD: <2.0Å
- TM-score: >0.70
- GDT_TS: >60

⏱️ **~6-8 hours** on A100  
💾 **Requires:** Colab Pro with A100

---

### 🧪 **Complete Production Benchmark**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/complete_production_run.ipynb)

**Full Quantum + Classical Benchmarking**

- Both quantum and classical training
- Statistical comparison
- Publication-ready analysis

⏱️ **Several hours** on A100

---

## 🎯 Quick Start Options

### 1. **Research Validation** (Free Colab) ⭐ **#1 RECOMMENDED**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_quantum_advantage_benchmark.ipynb)

**Publication-quality statistical testing** - Ready for papers!

### 2. **Beautiful Visualizations** (Free Colab) ⭐ **#2 RECOMMENDED**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/03_atomic_visualization_showcase.ipynb)

**World-class figures** - Perfect for presentations!

### 3. **Quick Demo** (Free Colab)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/colab_quickstart.ipynb)

**5-minute introduction** - See it in action

### 4. **Production Training** (Colab Pro A100)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_a100_ULTIMATE_MAXIMIZED.ipynb)

**AlphaFold2-level results** - Ultimate performance

### 5. Docker (Local)
```bash
docker-compose up
# Access JupyterLab at http://localhost:8888
```

---

## 🎯 Key Features

### Quantum Computing
- **Advanced Quantum Circuits**: Hardware-efficient ansatz with multiple entanglement topologies
- **Barren Plateau Mitigation**: Parameter scaling and Haar-random initialization
- **Quantum Kernels**: Sequence similarity in Hilbert space
- **Hybrid Architecture**: Gated fusion of quantum and classical representations
- **Noise Simulation**: Depolarizing noise for realistic quantum device modeling

### Deep Learning Architecture
- **Invariant Point Attention (IPA)**: Rotation and translation equivariant attention
- **Structure Module**: Iterative refinement with 8-12 layers of geometric reasoning
- **Pre-trained Embeddings**: ESM-2 (Meta AI) and ProtT5 (Rostlab) integration
- **Evolutionary Features**: PSSM, conservation scores, co-evolution matrices
- **Confidence Prediction**: pLDDT-style per-residue confidence scores

### Advanced Training
- **FAPE Loss**: Frame Aligned Point Error from AlphaFold
- **Mixed Precision**: FP16/BF16 training with automatic scaling
- **Exponential Moving Average (EMA)**: Stable weight averaging
- **Cosine Annealing**: Learning rate scheduling with warmup
- **Gradient Management**: Clipping and accumulation for stable training

### Statistical Validation
- **Hypothesis Testing**: Paired Wilcoxon and t-tests
- **Effect Sizes**: Cohen's d and rank-biserial correlation
- **Bootstrap CI**: 10,000-sample confidence intervals
- **Multiple Comparison Correction**: Bonferroni and Benjamini-Hochberg FDR
- **Power Analysis**: Statistical power calculation and sample size estimation

### World-Class Visualization
- **Interactive 3D**: py3Dmol molecular viewer
- **Structural Analysis**: Ramachandran plots, contact maps
- **Model Internals**: Attention heatmaps, quantum circuits
- **Publication Figures**: Multi-panel high-resolution outputs
- **Animations**: Structure refinement trajectories

---

## 🔬 Research Methodology

This project implements a rigorous experimental framework for evaluating quantum advantage:

### Evaluation Metrics
- **TM-score**: Template Modeling score (Zhang-Skolnick algorithm)
- **RMSD**: Root Mean Square Deviation with Kabsch alignment
- **GDT-TS/GDT-HA**: Global Distance Test scores
- **lDDT**: Local Distance Difference Test
- **pLDDT**: Per-residue confidence (0-100)
- **Contact Precision**: Residue-residue contact accuracy

### Quantum Advantage Testing
**Paired Comparison Protocol:**
1. Train quantum-enhanced model on protein dataset
2. Train identical classical baseline (quantum layers disabled)
3. Evaluate both on held-out CASP test set
4. Apply paired statistical tests (Wilcoxon, t-test)
5. Compute effect sizes (Cohen's d) and confidence intervals
6. Correct for multiple comparisons (Bonferroni/FDR)

**Statistical Validation:**
```python
from src.benchmarks.research_metrics import ResearchMetrics

metrics = ResearchMetrics()
results = metrics.compare_methods(
    quantum_scores=quantum_tm_scores,
    classical_scores=classical_tm_scores,
    metric_name='TM-score',
    higher_is_better=True
)
# p-values, effect sizes, CI, power analysis
```

---

## 🚀 Installation

### Option 1: Local Installation
```bash
git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
cd QuantumFold-Advantage
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .[dev,protein-lm]
pytest tests/
```

### Option 2: Docker (Recommended for Production)
```bash
docker-compose up -d
# JupyterLab: http://localhost:8888
# TensorBoard: http://localhost:6006
```

### Basic Usage
```python
import torch
from src.advanced_model import AdvancedProteinFoldingModel
from src.protein_embeddings import ESM2Embedder

embedder = ESM2Embedder(model_name='esm2_t33_650M_UR50D')
model = AdvancedProteinFoldingModel(
    input_dim=1280,
    c_s=384,
    c_z=128,
    use_quantum=True
)

sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
embeddings = embedder([sequence])

with torch.no_grad():
    output = model(embeddings['embeddings'])
    coordinates = output['coordinates']
    plddt = output['plddt']

print(f"Mean confidence: {plddt.mean():.3f}")
```

---

## 📁 Repository Structure

```
QuantumFold-Advantage/
├── src/
│   ├── quantum_layers.py              # Quantum circuits & hybrid layers
│   ├── advanced_model.py              # IPA, Structure Module, Confidence
│   ├── advanced_training.py           # FAPE, mixed precision, EMA
│   ├── protein_embeddings.py          # ESM-2, ProtT5, evolutionary features
│   ├── benchmarks/
│   │   └── research_metrics.py        # TM-score, RMSD, GDT, statistics
│   ├── visualization/
│   │   └── atomic_viz.py              # 3D, Ramachandran, contact maps
│   ├── data/
│   │   └── casp_loader.py             # CASP14/15, PDB, AlphaFold DB
│   └── utils/
│       ├── data_loader.py             # Dataset utilities
│       └── visualization.py           # Plotting functions
├── examples/
│   ├── 02_quantum_advantage_benchmark.ipynb    # ⭐ Statistical validation
│   ├── 03_atomic_visualization_showcase.ipynb  # ⭐ Publication figures
│   ├── 02_a100_ULTIMATE_MAXIMIZED.ipynb        # Production training
│   ├── 02_a100_production.ipynb                # A100 pipeline
│   ├── complete_production_run.ipynb           # Full benchmark
│   ├── colab_quickstart.ipynb                  # Quick demo
│   ├── 01_getting_started.ipynb                # Tutorial
│   ├── 02_quantum_vs_classical.ipynb           # Comparison
│   └── complete_benchmark.ipynb                # 30-60 min pipeline
├── tests/
│   ├── test_quantum_layers.py
│   ├── test_model.py
│   ├── test_benchmarks.py
│   └── test_visualization.py
├── .github/workflows/
│   ├── ci.yml                         # Continuous Integration
│   ├── test-notebooks.yml             # Comprehensive notebook testing
│   └── docker-publish.yml             # Docker CI/CD
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 🧪 Complete Notebook Catalog

### ⭐ Research-Grade (Publication Ready)
| Notebook | Description | Runtime | GPU | Colab |
|----------|-------------|---------|-----|-------|
| **02_quantum_advantage_benchmark** | Statistical validation, CASP15, LaTeX tables | 45 min | Free T4 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_quantum_advantage_benchmark.ipynb) |
| **03_atomic_visualization_showcase** | Interactive 3D, publication figures | 30 min | Free T4 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/03_atomic_visualization_showcase.ipynb) |

### 🔥 Production Training (Colab Pro)
| Notebook | Params | Data | Runtime | GPU | Colab |
|----------|--------|------|---------|-----|-------|
| **02_a100_ULTIMATE_MAXIMIZED** | 200M | CASP+RCSB | 10-12 hrs | A100 High RAM | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_a100_ULTIMATE_MAXIMIZED.ipynb) |
| **02_a100_production** | 85M | 5K proteins | 6-8 hrs | A100 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_a100_production.ipynb) |
| **complete_production_run** | - | CASP | 2-4 hrs | A100 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/complete_production_run.ipynb) |

### 🎓 Learning & Quick Start (Free Colab)
| Notebook | Description | Runtime | Colab |
|----------|-------------|---------|-------|
| **colab_quickstart** | 5-minute demo | 5 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/colab_quickstart.ipynb) |
| **01_getting_started** | Complete tutorial | 15-20 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/01_getting_started.ipynb) |
| **complete_benchmark** | Synthetic training pipeline | 30-60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/complete_benchmark.ipynb) |
| **02_quantum_vs_classical** | Comparison analysis | 20-30 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_quantum_vs_classical.ipynb) |

**📖 See [examples/README.md](examples/README.md) for detailed notebook guide**

---

## 📚 Key References

1. **AlphaFold-3** - Abramson et al., *Nature* (2024) DOI: 10.1038/s41586-024-07487-w
2. **ESM-2** - Lin et al., *Science* (2023) DOI: 10.1126/science.ade2574
3. **Quantum ML** - Schuld et al., *Phys. Rev. Lett.* (2019) DOI: 10.1103/PhysRevLett.122.040504
4. **FAPE Loss** - Jumper et al., *Nature* (2021) DOI: 10.1038/s41586-021-03819-2
5. **TM-score** - Zhang & Skolnick, *Proteins* (2004) DOI: 10.1002/prot.20264

---

## 🎓 Citation

```bibtex
@software{quantumfold2026,
  author = {Marena, Tommaso R.},
  title = {QuantumFold-Advantage: Quantum-Classical Hybrid Architecture for Protein Structure Prediction},
  year = {2026},
  institution = {The Catholic University of America},
  url = {https://github.com/Tommaso-R-Marena/QuantumFold-Advantage}
}
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Tommaso R. Marena**  
Undergraduate Researcher  
The Catholic University of America  
📧 marena@cua.edu  
🔗 [GitHub](https://github.com/Tommaso-R-Marena)

---

## 🙏 Acknowledgments

- Meta AI for ESM-2
- DeepMind for AlphaFold architecture
- Xanadu for PennyLane framework
- The protein structure prediction community

---

## 🔮 Future Directions

- [ ] Benchmark on CASP16 targets
- [ ] Multi-chain complex prediction
- [ ] RNA structure prediction
- [ ] Protein-ligand docking
- [ ] Real quantum device deployment
- [ ] Web API and cloud deployment

---

**⭐ Star this repository if you find it useful!**

**🔬 Run the [statistical validation](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_quantum_advantage_benchmark.ipynb) for publication-ready results!**

**🎨 Create [beautiful visualizations](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/03_atomic_visualization_showcase.ipynb) for presentations!**

**🚀 Train [production models](https://colab.research.google.com/github/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/examples/02_a100_ULTIMATE_MAXIMIZED.ipynb) for best results!**
## Quantum Code Discovery (Phase-1)

A new exploratory module is available under `src/quantum_codes_discovery/` with:
- graph-based stabilizer construction over GF(2),
- commutation checks and logical-operator extraction,
- SQLite storage for discovered candidates,
- Monte Carlo logical-error sweep utilities.

Run locally with:
```bash
./scripts/run_quantum_code_discovery.sh
```
