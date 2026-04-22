# Benchmark Run — 2026-04-22

## Request Context
This run was executed to test whether QuantumFold can outperform a classical baseline in at least one metric.

A **true AlphaFold 3 head-to-head benchmark is not possible in this repository alone** because AlphaFold 3 model weights/inference pipeline are not bundled here.

## Command Executed

```bash
python scripts/run_experiment.py \
  --n-proteins 24 --max-len 48 --d-model 48 --d-pair 24 \
  --n-heads 4 --n-evoformer 1 --n-struct-iter 1 \
  --n-qubits 2 --n-circuit-layers 1 --batch-size 4 --epochs 4 \
  --output-dir results/user_request_benchmark
```

## Key Results
- RMSD: Quantum 9.4363 vs Classical 9.4641 (**Quantum better; lower is better**)
- lDDT: Quantum 0.1891 vs Classical 0.1832 (**Quantum better; higher is better**)
- TM-score: Quantum 0.0922 vs Classical 0.0929 (Classical better)
- GDT-TS: Quantum 0.1400 vs Classical 0.1512 (Classical better)
- GDT-HA: Quantum 0.0221 vs Classical 0.0260 (Classical better)

Statistical significance was not established after Holm–Bonferroni correction.

## Interpretation
QuantumFold did better than the local classical baseline on some metrics (RMSD and lDDT) in this small synthetic-data run, but this should be treated as exploratory.

For a true AlphaFold 3 benchmark, this project needs:
1. A common real dataset (e.g., held-out CASP targets with released structures).
2. AF3 predictions generated under reproducible settings.
3. Matching post-processing and metric computation across methods.
