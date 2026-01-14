# Experimental Protocol: Quantum Advantage in Protein Structure Prediction

## Overview

This document outlines the rigorous experimental protocol for evaluating quantum advantage in protein structure prediction using the QuantumFold-Advantage framework.

**Protocol Version:** 1.0  
**Last Updated:** January 13, 2026  
**Author:** Tommaso R. Marena

---

## 1. Research Question

**Primary Hypothesis:** Quantum-enhanced neural networks demonstrate statistically significant improvements over classical baselines in protein structure prediction accuracy, as measured by TM-score, GDT-TS, and lDDT metrics.

**Null Hypothesis (H₀):** There is no significant difference in prediction accuracy between quantum-enhanced and classical models.

**Alternative Hypothesis (H₁):** Quantum-enhanced models achieve higher prediction accuracy than classical models.

---

## 2. Dataset Selection

### 2.1 Training Dataset

**Source:** CATH S35 domain database  
**Size:** 500-1000 diverse protein domains  
**Selection Criteria:**
- Sequence length: 50-300 residues
- Resolution: ≤ 2.5 Å
- R-factor: ≤ 0.25
- Structural diversity: Representatives from all CATH topology classes
- Sequence identity: ≤ 35% pairwise similarity

**Justification:** CATH S35 provides a non-redundant, structurally diverse dataset that covers the known protein fold space, ensuring model generalization.

### 2.2 Validation Dataset

**Source:** CATH S35 (separate split)  
**Size:** 15% of total dataset  
**Purpose:** Hyperparameter tuning and model selection

### 2.3 Test Dataset

**Primary Test Set:**
- **Source:** CATH S35 (held-out split)
- **Size:** 15% of total dataset
- **Purpose:** Final performance evaluation

**External Test Sets:**
1. **CASP16 FM targets** (Free Modeling - no templates)
   - Blind prediction targets from CASP16 competition
   - Size: 20-40 targets
   - Purpose: Assess generalization to unseen folds

2. **CAMEO dataset** (weekly blind predictions)
   - Recent PDB releases not in training data
   - Size: 50-100 structures
   - Purpose: Real-world validation

3. **Membrane proteins** (specific challenge)
   - Source: PDBTM or OPM databases
   - Size: 30-50 structures
   - Purpose: Assess performance on underrepresented class

---

## 3. Model Configurations

### 3.1 Quantum-Enhanced Model

**Architecture:**
```python
QuantumFoldModel(
    # Embedding
    embedding_model='esm2_t33_650M_UR50D',
    embedding_dim=1280,
    
    # Quantum layer
    use_quantum=True,
    n_qubits=8,
    n_quantum_layers=3,
    entanglement='circular',
    quantum_dropout=0.1,
    noise_model='depolarizing',
    noise_prob=0.01,
    
    # Classical architecture
    hidden_dim=512,
    n_transformer_layers=4,
    n_heads=8,
    n_structure_layers=8,
    
    # Outputs
    confidence_head=True
)
```

**Training Configuration:**
- Optimizer: AdamW (lr=5e-4, weight_decay=0.01)
- Scheduler: Cosine annealing with warmup (1000 steps)
- Loss: 5.0×FAPE + 3.0×MSE + 2.0×Distance + 0.5×Confidence
- Batch size: 1 (with gradient accumulation=4)
- Mixed precision: FP16
- Gradient clipping: 1.0
- EMA: α=0.999
- Total steps: 20,000
- Validation frequency: Every 1000 steps

### 3.2 Classical Baseline

**Architecture:** Identical to quantum model with `use_quantum=False`

**Critical Requirement:** All other hyperparameters must be identical to ensure fair comparison. The only difference is the presence/absence of quantum layers.

### 3.3 Additional Baselines

1. **ESM-Fold baseline:** Direct ESM-2 folding (transfer learning)
2. **Random initialization:** Same architecture, random weights
3. **Ablation controls:** See Section 6

---

## 4. Training Protocol

### 4.1 Randomization

**Random Seeds:** Set at multiple levels for reproducibility
```python
PYTHONHASHSEED=0
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True
```

**Data Splits:** Stratified by:
- Protein length (bins: 50-100, 100-150, 150-200, 200-300)
- CATH topology class (mainly-alpha, mainly-beta, alpha-beta, few-SS)

### 4.2 Training Execution

**For each model configuration:**

1. Initialize model with random seed
2. Load pre-computed ESM-2 embeddings (disk-cached)
3. Train for 20,000 steps
4. Save checkpoints every 1000 steps
5. Validate on validation set
6. Early stopping: patience=5 epochs on validation TM-score
7. Save best model based on validation performance

**Computational Requirements:**
- GPU: NVIDIA A100 (40GB) or T4 (15GB)
- Training time: ~45-60 minutes (T4), ~20-30 minutes (A100)
- Storage: ~50GB (embeddings) + ~2GB (checkpoints)

### 4.3 Reproducibility Checklist

- [ ] Git commit hash recorded
- [ ] Random seeds set
- [ ] Environment dependencies frozen (requirements.txt)
- [ ] Data split indices saved
- [ ] Training hyperparameters logged
- [ ] Model checkpoints saved
- [ ] Training curves exported

---

## 5. Evaluation Protocol

### 5.1 Metrics

**Primary Metrics:**
1. **TM-score** (0-1, higher better) - Global structural similarity
2. **GDT-TS** (0-100, higher better) - Fraction of residues within distance cutoffs
3. **lDDT** (0-1, higher better) - Local distance difference test

**Secondary Metrics:**
1. **RMSD** (Ć, lower better) - Root mean square deviation
2. **GDT-HA** (0-100, higher better) - High-accuracy GDT
3. **Contact precision/recall** - Residue-residue contact prediction
4. **pLDDT** (0-100, higher better) - Predicted confidence score

**Computational Cost Metrics:**
1. Inference time (seconds per protein)
2. Memory usage (GB)
3. FLOPs per prediction

### 5.2 Evaluation Procedure

**For each test protein:**

1. Load sequence and true structure
2. Generate ESM-2 embedding
3. Forward pass through model
4. Extract predicted coordinates
5. Kabsch superimposition for alignment
6. Compute all metrics
7. Record per-protein results

**Output:** CSV file with columns:
```
protein_id,tm_score,gdt_ts,gdt_ha,rmsd,lddt,contact_f1,plddt,length,topology_class
```

### 5.3 Visualization

**Required Figures:**

1. **Distribution plots** (violin/box plots)
   - TM-score distributions (quantum vs classical)
   - GDT-TS distributions
   - Stratified by protein length and topology

2. **Scatter plots**
   - Quantum vs Classical TM-score (per protein)
   - Above/below diagonal indicates quantum advantage
   - Color-code by protein length

3. **Training curves**
   - Validation TM-score vs training steps
   - Quantum and classical on same plot

4. **Confidence calibration**
   - pLDDT vs actual accuracy
   - Assess model uncertainty quantification

5. **3D structure examples**
   - Best cases (high TM-score)
   - Worst cases (low TM-score)
   - Side-by-side: true / quantum / classical

---

## 6. Statistical Analysis

### 6.1 Primary Analysis

**Paired Comparison Test:**

Both models predict the same test proteins, enabling paired statistical tests.

**Tests to Apply:**

1. **Wilcoxon Signed-Rank Test** (primary, non-parametric)
   - H₀: median(quantum - classical) = 0
   - H₁: median(quantum - classical) > 0
   - One-sided test, α = 0.05

2. **Paired t-test** (secondary, parametric)
   - Assumes normal distribution of differences
   - One-sided test, α = 0.05

3. **Sign test** (tertiary, most conservative)
   - Counts quantum wins vs classical wins

**Multiple Comparison Correction:**
- Testing multiple metrics (TM-score, GDT-TS, lDDT)
- Apply Benjamini-Hochberg FDR correction
- Report both raw and corrected p-values

### 6.2 Effect Size Calculation

**Metrics:**

1. **Cohen's d** (standardized mean difference)
   $$d = \frac{\bar{x}_{quantum} - \bar{x}_{classical}}{s_{pooled}}$$
   
   Interpretation:
   - |d| < 0.2: negligible
   - 0.2 ≤ |d| < 0.5: small
   - 0.5 ≤ |d| < 0.8: medium
   - |d| ≥ 0.8: large

2. **Rank-biserial correlation** (effect size for Wilcoxon)
   $$r = \frac{n_{wins} - n_{losses}}{n_{total}}$$

3. **Cliff's Delta** (non-parametric effect size)
   Robust to outliers and skewed distributions

### 6.3 Confidence Intervals

**Bootstrap CIs (10,000 samples):**
- Mean difference in TM-score
- Median difference in TM-score
- 95% confidence intervals

**Interpretation:**
- If CI does not include 0 → significant difference
- Report CI width (precision of estimate)

### 6.4 Power Analysis

**Post-hoc power calculation:**
- Given: observed effect size, sample size, α=0.05
- Compute: achieved power (1 - β)

**Minimum power threshold:** 0.80 (80%)

**If power < 0.80:**
- Report required sample size for 80% power
- Acknowledge limitation
- Plan follow-up experiments

### 6.5 Stratified Analysis

**Subgroup Analyses:**

1. **By protein length:**
   - Short (50-100 residues)
   - Medium (100-200 residues)
   - Long (200-300 residues)
   - Test for interaction effects

2. **By structural class:**
   - All-alpha
   - All-beta
   - Alpha-beta
   - Test if quantum advantage is fold-specific

3. **By prediction difficulty:**
   - Easy (classical TM > 0.7)
   - Medium (0.5 < classical TM < 0.7)
   - Hard (classical TM < 0.5)
   - Assess if quantum helps on hard cases

---

## 7. Ablation Studies

**Purpose:** Isolate the contribution of quantum components.

### 7.1 Quantum Architecture Ablations

| Ablation | Description | Purpose |
|----------|-------------|----------|
| No quantum | Baseline | Control |
| Linear entanglement | Change entanglement topology | Test entanglement effect |
| Circular entanglement | Default configuration | Main model |
| All-to-all entanglement | Maximum entanglement | Upper bound |
| 1 quantum layer | Reduce depth | Test layer contribution |
| 3 quantum layers | Default | Main model |
| 6 quantum layers | Increase depth | Test scaling |
| No noise | Ideal quantum | Upper bound |
| 1% depolarizing noise | Realistic NISQ | Main model |
| 5% depolarizing noise | Noisy NISQ | Robustness test |

### 7.2 Statistical Comparison

For each ablation:
- Train 3 independent runs (different seeds)
- Evaluate on same test set
- Paired comparison to baseline
- Report mean ± std across runs

---

## 8. Quantum Computational Advantage Analysis

### 8.1 Classical Simulability

**Test:** Can classical model match quantum performance?

**Procedure:**
1. Increase classical model capacity:
   - Add more layers (6 → 8 → 12)
   - Increase hidden dim (512 → 768 → 1024)
   - Add more attention heads (8 → 12 → 16)

2. Train until convergence

3. Compare to quantum model:
   - If classical matches: no quantum advantage (capacity issue)
   - If classical fails: potential quantum advantage

### 8.2 Quantum Resource Scaling

**Analysis:**
- Plot TM-score vs n_qubits (4, 6, 8, 10, 12)
- Plot TM-score vs n_quantum_layers (1, 2, 3, 4, 5)
- Identify saturation points

**Expected Behavior:**
- If quantum: performance improves with quantum resources
- If classical: performance plateaus quickly

### 8.3 Entanglement Witness

**Theoretical Analysis:**
- Compute von Neumann entropy of quantum states
- Measure entanglement in learned circuits
- Correlate entanglement with prediction accuracy

**Hypothesis:** Higher entanglement correlates with better performance on hard proteins.

---

## 9. Reporting Standards

### 9.1 Required Information

**Methods Section:**
- Complete model architecture (include diagram)
- Training procedure (algorithms, hyperparameters)
- Dataset details (sources, sizes, splits)
- Computational resources (hardware, time)
- Statistical methods (tests, corrections, CI)
- Code availability (GitHub repo + version/commit)

**Results Section:**
- Summary statistics (mean ± std for all metrics)
- Statistical test results (test statistic, p-value, effect size)
- Confidence intervals (bootstrap 95% CI)
- Figures (required plots from Section 5.3)
- Tables (per-protein results in supplementary)

**Supplementary Materials:**
- Complete per-protein results (CSV)
- Training curves (all runs)
- Ablation study results
- Hyperparameter sensitivity analysis
- Failed experiments and negative results

### 9.2 Checklist for Publication

**Reproducibility:**
- [ ] Code publicly available (GitHub)
- [ ] Pre-trained models shared (HuggingFace/Zenodo)
- [ ] Training data described (or shared if allowed)
- [ ] Random seeds reported
- [ ] Compute requirements specified
- [ ] Environment fully specified (Docker image)

**Statistical Rigor:**
- [ ] Multiple comparison correction applied
- [ ] Effect sizes reported (not just p-values)
- [ ] Confidence intervals provided
- [ ] Power analysis conducted
- [ ] Assumptions checked (normality, independence)
- [ ] Negative results included

**Transparency:**
- [ ] All hyperparameters reported
- [ ] Model selection procedure described
- [ ] Failed attempts documented
- [ ] Limitations discussed
- [ ] Conflicts of interest declared

---

## 10. Timeline and Milestones

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Data preparation | 1 week | Dataset splits + embeddings cached |
| 2. Baseline training | 1 week | Classical model trained + evaluated |
| 3. Quantum training | 1 week | Quantum model trained + evaluated |
| 4. Ablation studies | 2 weeks | All ablations completed |
| 5. Statistical analysis | 1 week | Hypothesis tests + effect sizes |
| 6. Figure generation | 1 week | Publication-ready figures |
| 7. Manuscript writing | 2 weeks | Complete draft |
| 8. Internal review | 1 week | Revised manuscript |
| **Total** | **10 weeks** | **Submitted manuscript** |

---

## 11. Expected Outcomes

### 11.1 Success Criteria

**Primary Success:**
- FDR-corrected p-value < 0.05 (Wilcoxon test)
- Cohen's d > 0.3 (small-to-medium effect)
- Mean TM-score improvement > 0.05
- Statistical power > 0.80

**Publication Tier:**
- Top-tier journal (Nature/Science/Cell): d > 0.8, p < 0.001, TM improvement > 0.15
- High-impact journal (Nature Methods/Commun): d > 0.5, p < 0.01, TM improvement > 0.10
- Domain journal (Proteins/Bioinformatics): d > 0.3, p < 0.05, TM improvement > 0.05

### 11.2 Negative Result Handling

**If no significant quantum advantage:**

1. Report honestly (important negative result)
2. Analyze why (capacity, training, architecture?)
3. Identify limitations
4. Propose improvements
5. Submit to journal accepting negative results

**Scientific value:** Negative results are crucial for the field.

---

## 12. References

1. Kryshtafovych et al., "Critical assessment of methods of protein structure prediction (CASP)—Round XIV", *Proteins* (2021)
2. Zhang & Skolnick, "Scoring function for automated assessment of protein structure template quality", *Proteins* (2004)
3. Mariani et al., "lDDT: a local superposition-free score for comparing protein structures and models", *Bioinformatics* (2013)
4. Cohen, J., "Statistical Power Analysis for the Behavioral Sciences" (1988)
5. Benjamini & Hochberg, "Controlling the False Discovery Rate", *J. Royal Stat. Soc.* (1995)
6. Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets", *JMLR* (2006)

---

## Appendix A: Code Execution Example

```python
import numpy as np
from src.benchmarks.casp_evaluation import CASPEvaluator
from src.analysis.quantum_advantage import QuantumAdvantageAnalyzer

# Initialize evaluators
casp_eval = CASPEvaluator(verbose=True)
qa_analyzer = QuantumAdvantageAnalyzer(n_bootstrap=10000, random_state=42)

# Load predictions
quantum_preds = load_predictions('quantum_model_predictions.pt')
classical_preds = load_predictions('classical_model_predictions.pt')
ground_truth = load_ground_truth('test_structures.pt')

# Evaluate structures
quantum_metrics = casp_eval.batch_evaluate(
    quantum_preds['coords'],
    ground_truth['coords'],
    quantum_preds['plddt']
)

classical_metrics = casp_eval.batch_evaluate(
    classical_preds['coords'],
    ground_truth['coords'],
    classical_preds['plddt']
)

# Extract TM-scores
quantum_tm = np.array([m.tm_score for m in quantum_metrics])
classical_tm = np.array([m.tm_score for m in classical_metrics])

# Statistical analysis
result = qa_analyzer.analyze_advantage(
    quantum_scores=quantum_tm,
    classical_scores=classical_tm,
    metric_name='TM-score',
    higher_is_better=True,
    alpha=0.05,
    n_comparisons=3  # Testing 3 metrics
)

# Report results
print(f"Quantum mean TM-score: {result.quantum_mean:.4f} ± {result.quantum_std:.4f}")
print(f"Classical mean TM-score: {result.classical_mean:.4f} ± {result.classical_std:.4f}")
print(f"Mean difference: {result.mean_difference:.4f}")
print(f"95% CI: [{result.mean_diff_ci_lower:.4f}, {result.mean_diff_ci_upper:.4f}]")
print(f"\nWilcoxon p-value: {result.wilcoxon_pvalue:.6f}")
print(f"FDR-corrected p-value: {result.fdr_pvalue:.6f}")
print(f"Cohen's d: {result.cohens_d:.3f}")
print(f"Statistical power: {result.statistical_power:.3f}")
print(f"\nSignificant quantum advantage: {result.is_significant()}")

# Save results
qa_analyzer.save_results(result, 'quantum_advantage_analysis.json')
```

---

**Document Status:** Final  
**Approved By:** Tommaso R. Marena  
**Date:** January 13, 2026
