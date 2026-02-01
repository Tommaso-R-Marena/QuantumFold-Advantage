# QuantumFold-Advantage Features Guide

Comprehensive guide to all features, modules, and capabilities.

---

## Table of Contents

- [Research Metrics](#research-metrics)
- [Visualization](#visualization)
- [Data Loading](#data-loading)
- [Quantum Layers](#quantum-layers)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Notebooks](#notebooks)

---

## Research Metrics

### TM-score (Template Modeling Score)

**Location:** `src/benchmarks/research_metrics.py`

**Purpose:** Measure structural similarity between predicted and true structures, normalized by protein length.

**Algorithm:**
1. Kabsch alignment (optimal rotation/translation)
2. Distance calculation for aligned residues
3. Length-normalized scoring (0-1 scale)

**Usage:**
```python
from src.benchmarks.research_metrics import ResearchMetrics

metrics = ResearchMetrics()
tm_score = metrics.calculate_tm_score(
    pred_coords,  # (N, 3) array
    true_coords,  # (N, 3) array
    sequence      # string
)
print(f"TM-score: {tm_score:.3f}")
# >0.5: same fold, >0.6: high confidence
```

**Interpretation:**
- **TM > 0.5**: Same overall fold
- **TM > 0.6**: High confidence in topology
- **TM > 0.7**: Very good model
- **TM > 0.8**: Near-native quality

---

### RMSD (Root Mean Square Deviation)

**Purpose:** Measure average distance between corresponding atoms after optimal alignment.

**Usage:**
```python
rmsd = metrics.calculate_rmsd(
    pred_coords,
    true_coords,
    align=True  # Kabsch alignment
)
print(f"RMSD: {rmsd:.2f} Å")
```

**Interpretation:**
- **RMSD < 1.5 Å**: Excellent
- **RMSD < 2.5 Å**: Very good
- **RMSD < 4.0 Å**: Good
- **RMSD > 4.0 Å**: Needs improvement

---

### GDT-TS (Global Distance Test - Total Score)

**Purpose:** Percentage of residues under multiple distance cutoffs.

**Usage:**
```python
gdt_ts = metrics.calculate_gdt_ts(pred_coords, true_coords)
print(f"GDT_TS: {gdt_ts:.1f}")
# Range: 0-100
```

**Calculation:** Average of residues within 1Å, 2Å, 4Å, 8Å

**Interpretation:**
- **GDT_TS > 70**: Excellent model
- **GDT_TS > 50**: Good model
- **GDT_TS < 30**: Poor model

---

### Statistical Validation

**Paired Comparison:**
```python
results = metrics.compare_methods(
    quantum_scores=[0.72, 0.75, 0.68, ...],
    classical_scores=[0.65, 0.70, 0.63, ...],
    metric_name='TM-score',
    higher_is_better=True
)

print(f"p-value (Wilcoxon): {results['wilcoxon_p']:.4f}")
print(f"Effect size (Cohen's d): {results['cohens_d']:.3f}")
print(f"95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
```

**Bootstrap Confidence Intervals:**
```python
ci_lower, ci_upper = metrics.bootstrap_confidence_interval(
    scores,
    n_bootstrap=10000,
    confidence=0.95
)
```

**LaTeX Table Generation:**
```python
latex = metrics.generate_latex_table(results)
print(latex)
# Ready to paste into paper!
```

---

## Visualization

### Interactive 3D Structures

**Location:** `src/visualization/atomic_viz.py`

**Basic Usage:**
```python
from src.visualization import ProteinVisualizer

viz = ProteinVisualizer(style='publication')

# Interactive 3D viewer
html = viz.visualize_3d_structure(
    coordinates,     # (N, 3) numpy array
    sequence,        # string
    confidence=plddt,  # (N,) array, optional
    secondary_structure='HHHEEECCC...',  # optional
    style='cartoon',   # or 'stick', 'sphere', 'surface'
    color_by='confidence',  # or 'secondary_structure', 'uniform'
    width=800,
    height=600
)

# Display in Jupyter
from IPython.display import HTML, display
display(HTML(html))
```

**Color Schemes:**
- `'confidence'`: AlphaFold pLDDT style (blue=high, red=low)
- `'secondary_structure'`: Pink=helix, yellow=sheet, cyan=coil
- `'uniform'`: Single color

---

### Ramachandran Plots

**Purpose:** Validate backbone geometry using phi-psi angles.

**Usage:**
```python
fig = viz.plot_ramachandran(
    coordinates,
    sequence,
    secondary_structure='HHHEEECC...',  # optional
    figsize=(10, 10)
)
plt.savefig('ramachandran.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Features:**
- Favorable/allowed region shading
- Secondary structure color coding
- Outlier detection
- Publication-quality formatting

---

### Contact Maps

**Purpose:** Visualize residue-residue proximity.

**Usage:**
```python
fig = viz.plot_contact_map(
    coordinates,
    sequence,
    threshold=8.0,  # Ångströms
    secondary_structure='HHH...',  # optional
    figsize=(12, 10)
)
plt.savefig('contact_map.png', dpi=300)
```

**Customization:**
- Distance thresholds
- Secondary structure annotations
- Color maps
- Diagonal masking

---

### Attention Heatmaps

**Purpose:** Visualize transformer attention patterns.

**Usage:**
```python
# attention: (num_heads, seq_len, seq_len)
fig = viz.plot_attention_heatmap(
    attention,
    sequence,
    layer_idx=0,
    head_idx=0,
    figsize=(14, 12)
)
```

---

### Quantum Circuit Diagrams

**Purpose:** Visualize quantum circuit architecture.

**Usage:**
```python
gate_sequence = ['H', 'RY', 'RZ', 'CNOT', ...]
fig = viz.plot_quantum_circuit(
    num_qubits=8,
    circuit_depth=4,
    gate_sequence=gate_sequence,
    figsize=(16, 8)
)
```

---

### Structure Refinement Animations

**Purpose:** Visualize iterative structure refinement.

**Usage:**
```python
# trajectory: (num_steps, seq_len, 3)
gif_path = viz.create_trajectory_animation(
    trajectory,
    sequence,
    output_path='refinement.gif',
    confidence=plddt,  # optional
    fps=5
)
```

**Output:** Animated GIF showing structure evolution

---

## Data Loading

### CASP Targets

**Location:** `src/data/casp_loader.py`

**Basic Usage:**
```python
from src.data.casp_loader import CASPDataLoader

loader = CASPDataLoader(casp_version=15)

# Get targets with filtering
targets = loader.get_targets(
    max_targets=10,
    min_length=50,
    max_length=500,
    difficulty_range=['medium', 'hard']
)

for target in targets:
    print(f"{target['id']}: {target['length']} residues")
    coords = target['coordinates']  # (N, 3)
    seq = target['sequence']         # string
    ss = target['secondary_structure']  # DSSP string
```

**Target Properties:**
- `id`: CASP target ID (e.g., 'T1124')
- `sequence`: Amino acid sequence
- `coordinates`: CA coordinates (N, 3)
- `length`: Number of residues
- `difficulty`: 'easy', 'medium', or 'hard'
- `secondary_structure`: DSSP string (H/E/C)

---

### Real PDB Structures

**Download from RCSB:**
```python
target_data = loader.download_real_structure(
    pdb_id='1ABC',
    target_id='custom_target'
)
```

**Download from AlphaFold DB:**
```python
target_data = loader.get_alphafold_structure(
    uniprot_id='P12345',
    target_id='alphafold_target'
)
# Includes pLDDT confidence scores
```

---

## Quantum Layers

### Quantum Feature Encoder

**Location:** `src/quantum_layers.py`

**Usage:**
```python
from src.quantum_layers import QuantumFeatureEncoder

encoder = QuantumFeatureEncoder(
    num_qubits=8,
    num_layers=4,
    entanglement='linear',  # or 'circular', 'all'
    noise_level=0.01
)

# Input: (batch, seq_len, features)
quantum_features = encoder(classical_features)
```

**Parameters:**
- `num_qubits`: Number of qubits in circuit
- `num_layers`: Circuit depth
- `entanglement`: Topology ('linear', 'circular', 'all')
- `noise_level`: Depolarizing noise probability

---

### Hybrid Quantum-Classical Layer

**Usage:**
```python
from src.quantum_layers import HybridQuantumClassicalLayer

hybrid = HybridQuantumClassicalLayer(
    input_dim=256,
    output_dim=256,
    num_qubits=8,
    fusion_strategy='gated'  # or 'concat', 'add'
)

output = hybrid(input_features)
```

---

## Model Architecture

### Advanced Protein Folding Model

**Location:** `src/advanced_model.py`

**Full Model:**
```python
from src.advanced_model import AdvancedProteinFoldingModel

model = AdvancedProteinFoldingModel(
    input_dim=1280,      # ESM-2 embedding size
    c_s=384,             # Single representation
    c_z=128,             # Pair representation
    use_quantum=True,    # Enable quantum layers
    num_qubits=8,
    num_encoder_layers=12,
    num_structure_layers=8,
    num_heads=12,
    dropout=0.1
)

# Forward pass
output = model(embeddings)

# Outputs:
coordinates = output['coordinates']  # (B, L, 3)
plddt = output['plddt']              # (B, L)
trajectory = output['trajectory']    # (B, num_iters, L, 3)
```

---

## Training Pipeline

### Advanced Training

**Location:** `src/advanced_training.py`

**Basic Training:**
```python
from src.advanced_training import AdvancedProteinTrainer

trainer = AdvancedProteinTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_amp=True,        # Mixed precision
    use_ema=True,        # Exponential moving average
    gradient_clip=1.0,
    use_wandb=True
)

trainer.train(
    num_epochs=100,
    save_dir='checkpoints',
    validate_every=5
)
```

**Loss Components:**
- FAPE (Frame Aligned Point Error)
- RMSD loss
- Confidence loss (pLDDT)
- Local geometry loss
- Perceptual loss

---

## Notebooks

### Research Validation

**File:** `examples/02_quantum_advantage_benchmark.ipynb`

**Purpose:** Publication-quality statistical validation

**Outputs:**
1. Statistical test results (p-values, effect sizes)
2. LaTeX tables for papers
3. Publication-ready figures
4. Power analysis report

**Runtime:** 30-45 minutes (free Colab T4)

---

### Visualization Showcase

**File:** `examples/03_atomic_visualization_showcase.ipynb`

**Purpose:** Create beautiful publication figures

**Outputs:**
1. Interactive 3D viewers (HTML)
2. Ramachandran plots (PNG, 300 DPI)
3. Contact maps (PNG, 300 DPI)
4. Attention heatmaps (PNG, 300 DPI)
5. Multi-panel figures (PNG, 300 DPI)
6. Refinement animations (GIF)

**Runtime:** 20-30 minutes (free Colab T4)

---

## Tips and Best Practices

### Performance Optimization

1. **Use mixed precision** (FP16/BF16) for faster training
2. **Enable gradient checkpointing** for large models
3. **Batch by sequence length** to minimize padding
4. **Cache embeddings** when possible
5. **Use DataLoader workers** (but set to 0 on Colab)

### Visualization

1. **Always use 300 DPI** for publications
2. **Save as SVG** for vector graphics when possible
3. **Use consistent color schemes** across figures
4. **Add scale bars** to structural visualizations
5. **Include confidence scores** when showing predictions

### Statistical Validation

1. **Use paired tests** when comparing methods
2. **Report effect sizes**, not just p-values
3. **Include confidence intervals** (bootstrap recommended)
4. **Correct for multiple comparisons**
5. **Check statistical power** before concluding

---

**Last Updated:** February 1, 2026
