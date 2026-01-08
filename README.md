# QuantumFold-Advantage

A compact, reproducible research scaffold implementing a hybrid quantum–classical dynamic protein-folding pipeline with rigorous experimental framework for evaluating whether a hybrid approach can outperform AlphaFold-3 on selected benchmarks.

**Important:** This repository implements a toy research framework. Do not claim to beat AlphaFold-3 unless `ADVANTAGE_CLAIM_PROTOCOL.md` requirements are met and benchmark scripts produce measured results meeting pre-registered statistical thresholds.

## Why Useful for STEM

Protein structure prediction is fundamental to understanding biological function, drug design, and disease mechanisms. While AlphaFold-3 achieves state-of-the-art accuracy, hybrid quantum–classical approaches offer potential advantages through enhanced optimization landscapes and novel feature representations. This repository provides:

- Complete experimental pipeline for rigorous hypothesis testing
- Public benchmark datasets with synthetic fallbacks
- Statistical evaluation framework (RMSD, TM-score, permutation tests)
- AlphaFold-style visualizations using py3Dmol
- Provenance tracking and reproducibility guarantees

## Scientific Rationale

**Protein Folding Challenge:** Predicting 3D protein structure from amino acid sequences remains computationally intensive. AlphaFold-3 uses deep learning with attention mechanisms, achieving TM-scores >0.8 for many targets.

**Hybrid Quantum Approach:** Variational quantum circuits (VQCs) can parameterize complex energy landscapes efficiently. PennyLane enables differentiable quantum–classical optimization. This work explores whether quantum feature encoding and entanglement improve small-protein structure prediction.

**Evaluation Metrics:** 
- RMSD (root-mean-square deviation) measures atomic position differences
- TM-score (template modeling score) accounts for global topology; values >0.5 indicate similar folds
- Statistical significance via paired permutation tests and bootstrap confidence intervals

## Quick Start

### Google Colab (Recommended)

1. Open `notebooks/colab_run.ipynb` in Google Colab
2. Run the bootstrap cell (installs requirements, detects GPU)
3. Execute all cells sequentially
4. Visualizations and results appear inline; outputs saved to `outputs/`

### Local Installation

```bash
git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
cd QuantumFold-Advantage
pip install -r requirements.txt
python run_demo.py
```

**CPU Fallback:** Code automatically detects missing GPU and adjusts batch size, epochs, and model capacity.

## Running Benchmarks

Benchmarks measure wall-clock time, memory, quantum resource usage, and accuracy:

```bash
python scripts/benchmark.py
```

Results written to:
- `outputs/benchmark_results.json` — timing, memory, RMSD/TM-score
- `outputs/benchmark_plot.png` — visual comparison

To run full quantum advantage experiments (paired classical vs. hybrid):

```bash
python scripts/quantum_experiment.py
python scripts/statistical_evaluation.py
```

See `docs/ADVANTAGE_CLAIM_PROTOCOL.md` for required thresholds and evidence.

## Repository Structure

```
QuantumFold-Advantage/
├── run_demo.py                          # Main entrypoint
├── requirements.txt                      # Pinned dependencies
├── LICENSE                               # MIT License
├── CITATION.md                           # Academic citations
├── notebooks/
│   └── colab_run.ipynb                  # Colab-ready notebook
├── scripts/
│   ├── fetch_and_verify.py              # Download + checksum + provenance
│   ├── benchmark.py                     # Environment detection + benchmarking
│   ├── quantum_experiment.py            # Paired hybrid vs. baseline orchestration
│   └── statistical_evaluation.py        # RMSD, TM-score, permutation tests, bootstrap CI
├── src/
│   ├── __init__.py
│   ├── data.py                          # Loader + synthetic fallback generator
│   ├── model.py                         # Classical + optional PennyLane hybrid layer
│   ├── train.py                         # Training loop + metrics
│   ├── pipeline.py                      # Orchestrator
│   ├── visualize.py                     # py3Dmol PDB writer + visualization
│   └── utils.py                         # Seed, device detection, logging
├── tests/
│   ├── __init__.py
│   ├── test_data.py                     # Data generation tests
│   └── test_smoke.py                    # Quick integration test
├── docs/
│   ├── ADVANTAGE_CLAIM_PROTOCOL.md      # Pre-registration, thresholds, evidence
│   ├── ETHICS_AND_CLAIMS.md             # Ethics, dual-use, responsible claims
│   ├── CLAIM_CHECKLIST.md               # Step-by-step verification
│   ├── LEGAL.md                         # Licenses, copyright, legal compliance
│   └── NOTES.md                         # Limitations and design decisions
├── outputs/
│   └── .gitkeep                         # Preserve directory structure
└── .github/
    └── workflows/
        └── ci.yml                       # GitHub Actions CI
```

## Data Sources

- **PDB (Protein Data Bank):** Small protein structures (1-50 residues) downloaded via RCSB API
- **Synthetic Fallback:** Generates random valid sequences + toy coordinates if network unavailable
- **Provenance:** All downloads logged in `outputs/sources.json` with URLs, DOIs, licenses, SHA256

## Reproducibility

- Python 3.10+
- All dependencies pinned in `requirements.txt`
- Random seeds fixed (`RANDOM_SEED=42` in code)
- Exact command: `python run_demo.py --seed 42 --mode cpu`

## Testing

```bash
pytest tests/ -v
```

GitHub Actions CI runs tests on `ubuntu-latest` with pinned dependencies.

## Citations

See `CITATION.md` for required academic references. Key citations:

- AlphaFold-3: Abramson et al., Nature 2024, DOI: 10.1038/s41586-024-07487-w
- TM-score: Zhang & Skolnick, NAR 2005, DOI: 10.1093/nar/gki524
- Quantum ML for proteins: Hirai et al., arXiv 2508.03446

## License

MIT License. See `LICENSE` file. All external data sources documented in `outputs/sources.json` with respective licenses.

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Update `CITATION.md` for new algorithms/datasets
3. Follow reproducibility guidelines in `docs/ADVANTAGE_CLAIM_PROTOCOL.md`

## Disclaimer

**This is a research scaffold, not a production tool.** Performance claims require rigorous statistical validation. Do not claim superiority over AlphaFold-3 unless all requirements in `docs/ADVANTAGE_CLAIM_PROTOCOL.md` are met.

## Contact

For questions about methodology or reproducibility, open a GitHub issue.