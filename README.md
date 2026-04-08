# QuantumFold-Advantage

[![CI](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/ci.yml/badge.svg)](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A hybrid quantum-classical framework for protein structure prediction that
rigorously evaluates whether variational quantum circuits can improve upon
purely classical deep learning baselines inspired by AlphaFold.

---

## Abstract

Protein structure prediction remains a central challenge in computational
biology with implications for drug discovery, genetic disease understanding,
and protein engineering.  While classical deep learning approaches such as
AlphaFold2 have achieved remarkable accuracy, the exponential growth of Hilbert
space dimensionality in quantum systems suggests potential advantages for
capturing long-range correlations in protein sequences.

**QuantumFold-Advantage** integrates hardware-efficient variational quantum
circuits with an AlphaFold-inspired neural architecture — Evoformer blocks,
Invariant Point Attention (IPA), and an iterative Structure Module — to produce
3D backbone coordinates from amino acid sequences.  A single `quantum_enabled`
flag controls whether quantum enhancement layers are active, enabling rigorous
ablation: the only difference between the two conditions is the presence of
quantum processing.

Evaluation follows CASP (Critical Assessment of protein Structure Prediction)
standards using RMSD, TM-score, GDT-TS, GDT-HA, and lDDT.  Statistical
validation employs paired bootstrap hypothesis tests, Wilcoxon signed-rank
tests, Cohen's d effect sizes, and Holm–Bonferroni multiple-testing correction.

---

## Architecture

```
Input: Amino acid sequence
         │
         ▼
┌──────────────────────────┐
│  Protein Embedding       │  Learned AA embedding + physicochemical features
│  + Positional Encoding   │  + sinusoidal position encoding
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Quantum Enhancement     │  Sliding-window VQC (PennyLane)
│  Layer  [if enabled]     │  with gated residual fusion
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Evoformer Stack         │  Self-attention → pair update → transition
│  (N blocks)              │  (inspired by AlphaFold2 Evoformer)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Quantum Attention       │  Quantum-modulated multi-head attention
│  [if enabled]            │  (one head replaced by VQC)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Structure Module        │  Iterative IPA refinement →
│  (K iterations)          │  backbone frames → atom coordinates
└──────────┬───────────────┘
           │
           ▼
Output: 3D backbone coordinates (N, Cα, C)
```

### Quantum Components

| Component | Purpose | Qubits | Circuit Depth |
|---|---|---|---|
| **Quantum Enhancement** | Sliding-window feature augmentation via VQC | 4–8 | 2–4 layers |
| **Quantum Attention** | Modulate attention weights with quantum kernels | 2–4 | 1–3 layers |
| **Barren Plateau Mitigation** | Haar-random init + per-layer gradient scaling | — | — |
| **Noise Simulation** | Depolarizing channels for hardware realism | — | — |

---

## Installation

```bash
git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
cd QuantumFold-Advantage
pip install pennylane torch numpy scipy matplotlib seaborn biopython pyyaml
```

### Quick Start

```bash
# Run a full experiment (synthetic data, both models, statistical comparison)
python scripts/run_experiment.py --epochs 20

# Classical-only baseline
python scripts/run_experiment.py --no-quantum --epochs 20

# Small demo (fast, ~2 min)
python scripts/run_experiment.py \
  --n-proteins 20 --max-len 32 --d-model 32 --d-pair 16 \
  --n-heads 2 --n-evoformer 1 --n-struct-iter 1 \
  --n-qubits 2 --n-circuit-layers 1 --epochs 5
```

### Run Tests

```bash
pip install pytest pytest-timeout
python -m pytest tests/test_hybrid_model.py -v
```

---

## Project Structure

```
QuantumFold-Advantage/
├── src/
│   ├── quantum_layers.py                 # Variational quantum circuits (VQC)
│   ├── models/
│   │   └── quantumfold_advantage.py      # Main hybrid model
│   ├── classical/
│   │   ├── attention.py                  # Multi-head self-attention + IPA
│   │   ├── evoformer.py                  # Evoformer stack
│   │   └── structure_module.py           # Iterative 3D coordinate prediction
│   ├── training/
│   │   ├── trainer.py                    # Training loop (dual-mode)
│   │   └── losses.py                     # FAPE, distance matrix, combined
│   ├── evaluation/
│   │   ├── metrics.py                    # RMSD, TM-score, GDT-TS, lDDT
│   │   ├── statistical_tests.py          # Bootstrap, Wilcoxon, Cohen's d
│   │   └── visualization.py             # Publication-quality plots
│   ├── data/
│   │   ├── hybrid_dataset.py             # PyTorch dataset + synthetic generator
│   │   └── casp_loader.py               # CASP data loading
│   └── statistical_tests.py             # Legacy statistical validation
├── scripts/
│   ├── run_experiment.py                 # Main experiment runner
│   ├── download_casp_data.py             # CASP data download utility
│   └── compare_models.py                # Model comparison pipeline
├── configs/
│   ├── quantum_enabled.yaml
│   ├── classical_only.yaml
│   └── default_config.yaml
├── tests/
│   └── test_hybrid_model.py              # 29 comprehensive tests
└── .github/workflows/ci.yml             # CI pipeline
```

---

## Evaluation Metrics

| Metric | Description | Range | Interpretation |
|---|---|---|---|
| **RMSD** | Root-mean-square deviation after Kabsch alignment | 0–∞ Å | Lower is better |
| **TM-score** | Template Modelling score (length-normalised) | (0, 1] | > 0.5 = same fold |
| **GDT-TS** | Global Distance Test at 1, 2, 4, 8 Å cutoffs | [0, 1] | Higher is better |
| **GDT-HA** | GDT at stricter 0.5, 1, 2, 4 Å cutoffs | [0, 1] | Higher is better |
| **lDDT** | Local Distance Difference Test | [0, 1] | Higher is better |

---

## Statistical Validation

The comparison pipeline applies:

1. **Paired bootstrap test** (10,000 resamples) for mean differences.
2. **Wilcoxon signed-rank test** (non-parametric paired test).
3. **Cohen's d** effect size with bootstrap confidence intervals.
4. **Holm–Bonferroni correction** for multiple comparisons across metrics.

---

## Configuration

See `configs/quantum_enabled.yaml` for the full parameter reference:

```yaml
model:
  d_model: 128          # Residue representation dimension
  n_heads: 8            # Attention heads
  n_evoformer_blocks: 4 # Evoformer depth
  quantum_enabled: true # Toggle quantum components

quantum:
  n_qubits: 8           # Qubits per circuit
  n_circuit_layers: 4   # VQC depth
  quantum_lr: 0.01      # Quantum parameter learning rate

training:
  epochs: 100
  lr: 0.0001            # Classical learning rate
  grad_clip: 1.0
  early_stopping_patience: 15
```

---

## References

1. Jumper, J. et al. "Highly accurate protein structure prediction with AlphaFold." *Nature* **596**, 583–589 (2021). [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

2. Casares, P. A. M., Campos, R. & Martin-Delgado, M. A. "QFold: Quantum Walks and Deep Learning to Solve Protein Folding." *Quantum Sci. Technol.* **7**, 025013 (2022). [doi:10.1088/2058-9565/ac4f2f](https://doi.org/10.1088/2058-9565/ac4f2f)

3. Outeiral, C. et al. "The prospects of quantum computing in computational molecular biology." *WIREs Comput Mol Sci* **11**, e1481 (2021). [doi:10.1002/wcms.1481](https://doi.org/10.1002/wcms.1481)

4. Raubenolt, B. et al. "A Perspective on Protein Structure Prediction Using Quantum Computers." *J. Chem. Theory Comput.* (2024). [doi:10.1021/acs.jctc.4c00067](https://doi.org/10.1021/acs.jctc.4c00067)

5. Geoffrey, A. S. "Protein structure prediction using AI and quantum computers." *bioRxiv* (2021). [doi:10.1101/2021.05.22.445242](https://doi.org/10.1101/2021.05.22.445242)

6. Emani, P. S. et al. "Quantum computing at the frontiers of biological sciences." *Nature Methods* **18**, 701–709 (2021). [doi:10.1038/s41592-020-01004-3](https://doi.org/10.1038/s41592-020-01004-3)

7. Zhang, Y. & Skolnick, J. "Scoring function for automated assessment of protein structure template quality." *Proteins* **57**, 702–710 (2004). [doi:10.1002/prot.20264](https://doi.org/10.1002/prot.20264)

---

## Citation

```bibtex
@software{marena2026quantumfold,
  author  = {Marena, Tommaso R.},
  title   = {QuantumFold-Advantage: A Hybrid Quantum-Classical Framework
             for Protein Structure Prediction},
  year    = {2026},
  url     = {https://github.com/Tommaso-R-Marena/QuantumFold-Advantage},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
