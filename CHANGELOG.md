# Changelog

All notable changes to QuantumFold-Advantage will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Major Feature Release - February 1, 2026

## Added

### Research Infrastructure

#### Publication-Quality Metrics (`src/benchmarks/research_metrics.py`)
- **TM-score calculation** with Zhang-Skolnick algorithm
  - Proper Kabsch superposition alignment
  - Rotation and translation optimization
  - Length normalization
- **RMSD computation** with optimal structural alignment
  - Kabsch algorithm implementation
  - CA-only and all-atom modes
- **GDT-TS and GDT-HA scores**
  - Multiple distance cutoffs (1, 2, 4, 8 Å)
  - Percentage of aligned residues
- **lDDT (Local Distance Difference Test)**
  - Local structure quality assessment
  - Multiple inclusion radii
- **Statistical validation framework**
  - Paired Wilcoxon signed-rank test
  - Paired t-test with proper assumptions
  - Effect size calculations (Cohen's d, rank-biserial correlation)
  - Bootstrap confidence intervals (10,000 samples)
  - Multiple comparison correction (Bonferroni, Benjamini-Hochberg FDR)
  - Statistical power analysis
  - Sample size estimation
- **LaTeX table generation** for publications
- **Publication-ready plotting** with customizable styles

#### World-Class Visualization (`src/visualization/atomic_viz.py`)
- **Interactive 3D molecular visualization**
  - py3Dmol integration for web-based viewing
  - Multiple rendering styles (cartoon, stick, sphere, surface)
  - Color schemes: confidence, secondary structure, custom
  - Rotation, zoom, and interactive controls
- **Ramachandran plot generation**
  - Phi-psi angle calculation from coordinates
  - Secondary structure overlay
  - Favorable/allowed region shading
  - Outlier detection
- **Contact map visualization**
  - Distance-based contact detection
  - Customizable distance thresholds
  - Secondary structure annotations
  - Heatmap with proper scaling
- **Attention mechanism heatmaps**
  - Multi-head attention visualization
  - Layer-wise analysis
  - Residue-residue interaction patterns
- **Quantum circuit diagrams**
  - Gate sequence visualization
  - Qubit wire representation
  - Entanglement topology display
- **Structure refinement animations**
  - Trajectory GIF generation
  - Frame-by-frame visualization
  - Confidence-colored trajectories
- **Multi-panel publication figures**
  - Customizable layouts
  - High-resolution output (300 DPI)
  - SVG vector graphics support
  - AlphaFold-style color schemes

#### CASP Data Infrastructure (`src/data/casp_loader.py`)
- **CASP14 and CASP15 target support**
  - Curated target lists by difficulty
  - Free modeling (FM) and template-based (TBM) targets
- **Real protein structure downloads**
  - RCSB PDB integration with automatic retrieval
  - AlphaFold Database support with pLDDT scores
  - Proper error handling and retries
- **Synthetic target generation**
  - Realistic amino acid frequency distributions
  - Constrained random walk for backbone geometry
  - Secondary structure element insertion (helices, sheets)
  - DSSP-style secondary structure strings
- **Caching system** for efficient data reuse
- **Filtering capabilities**
  - Length-based filtering
  - Difficulty-based selection
  - Domain classification

### Research-Grade Notebooks

#### Statistical Validation Notebook (`examples/02_quantum_advantage_benchmark.ipynb`)
- **Rigorous quantum vs. classical comparison**
  - Paired experimental design
  - Matched architectures (quantum layers on/off)
- **Real CASP15 benchmark targets**
  - Diverse difficulty levels
  - Standard evaluation protocol
- **Comprehensive statistical analysis**
  - Wilcoxon signed-rank test (non-parametric)
  - Paired t-test (parametric)
  - Bootstrap confidence intervals (10K samples)
  - Effect size calculations
  - Power analysis with recommendations
- **Publication-ready outputs**
  - LaTeX tables ready for papers
  - High-quality figures (300 DPI)
  - Statistical reporting following APA guidelines
- **Runtime:** 30-45 minutes on free Colab T4 GPU
- **Use cases:** Research papers, grant proposals, thesis work

#### Visualization Showcase Notebook (`examples/03_atomic_visualization_showcase.ipynb`)
- **Interactive 3D molecular structures**
  - Web-based py3Dmol viewer
  - Multiple viewing modes and styles
  - Confidence and secondary structure coloring
- **Structural analysis plots**
  - Ramachandran plots with secondary structure
  - Contact maps with annotations
  - Distance distribution analysis
- **Model internals visualization**
  - Attention mechanism heatmaps
  - Multi-head attention patterns
  - Layer-wise analysis
- **Quantum circuit diagrams**
  - Hardware-efficient ansatz visualization
  - Gate-level detail
  - Entanglement pattern display
- **Confidence visualization**
  - Per-residue pLDDT scores
  - AlphaFold-style color schemes
  - Statistical distribution plots
- **Multi-panel publication figures**
  - 4-6 panel layouts
  - Consistent formatting
  - Print-ready resolution
- **Animation generation**
  - Structure refinement trajectories
  - GIF output for presentations
  - Frame-by-frame control
- **Runtime:** 20-30 minutes on free Colab T4 GPU
- **Use cases:** Publications, presentations, posters, talks

### Testing Infrastructure

#### Comprehensive Notebook Testing Workflow (`.github/workflows/test-notebooks.yml`)
- **Multi-stage validation pipeline**
  1. **validate-notebooks**: JSON structure, metadata, common issues
  2. **syntax-check**: Python syntax validation via AST parsing
  3. **import-check**: Import statement extraction and verification
  4. **metadata-check**: Colab compatibility and GPU settings
  5. **dry-run-execution**: Limited cell execution (first 5 cells)
  6. **execute-lightweight**: Full execution of quickstart notebooks
  7. **test-summary**: Aggregated results and GitHub summary
- **Multi-version Python testing** (3.8, 3.9, 3.10)
- **Automated triggers**
  - Every push to main/develop
  - Every pull request
  - Weekly scheduled runs
  - Manual workflow dispatch
- **Security checks**
  - Hardcoded path detection
  - Credential exposure scanning
  - Excessive output warnings
- **Performance monitoring**
  - Execution time tracking
  - Timeout protection
  - Resource usage reporting

#### Pytest Notebook Test Suite (`tests/test_notebooks.py`)
- **100+ individual test cases** across 8 test classes
- **TestNotebookStructure**: JSON validity, cell presence, cell types
- **TestPythonCode**: Syntax, hardcoded paths, credentials, debugging
- **TestNotebookMetadata**: Metadata completeness, Colab compatibility
- **TestImports**: Import extraction, src module verification
- **TestNotebookContent**: Title presence, documentation ratio
- **TestExecutionOrder**: Sequential execution, count gaps
- **TestNotebookSize**: File size, cell count limits
- **TestOutputs**: Large output detection, git cleanliness
- **Parameterized testing** across all 14 notebooks
- **Detailed error reporting** with cell-level precision

### Documentation

#### Updated Main README (`README.md`)
- **Removed all non-existent file references**
  - No broken links
  - Accurate file paths
  - Current notebook inventory
- **Highlighted new research-grade notebooks**
  - Top billing for statistical validation
  - Featured visualization showcase
  - Clear use case guidance
- **Reorganized structure**
  - Research-grade section at top
  - Production training clearly separated
  - Learning resources well-organized
- **Complete notebook catalog**
  - Runtime estimates
  - Hardware requirements
  - Direct Colab links
  - Use case recommendations
- **Professional formatting**
  - Consistent emoji usage
  - Clear hierarchy
  - Scannable sections

#### Examples README (`examples/README.md`)
- **Comprehensive notebook guide**
  - Best notebooks to start with
  - Complete catalog by category
  - Detailed feature lists
- **Hardware requirement matrix**
  - Free Colab vs. Colab Pro
  - GPU specifications
  - RAM requirements
- **Use case decision tree**
  - "I want to..." scenarios
  - Direct recommendations
  - Expected outcomes
- **Troubleshooting section**
  - Common issues and solutions
  - Performance tips
  - Debugging guidance

#### Integration Summary (`INTEGRATION_SUMMARY.md`)
- **Complete change documentation**
  - All new files listed
  - Detailed feature descriptions
  - Usage examples for each component
- **Quality checklist**
  - Code quality verification
  - Testing coverage
  - Documentation completeness
  - Scientific rigor
- **Graduate school readiness assessment**
  - Technical depth demonstration
  - Scientific communication
  - Software engineering practices
  - Research excellence indicators

## Changed

### Infrastructure Improvements
- **CI/CD pipeline** now includes comprehensive notebook testing
- **Test coverage** expanded to include all notebooks
- **Documentation** completely revised for accuracy
- **Repository structure** clearly documented

### Notebook Updates
- All notebooks verified for structural integrity
- Metadata validated for Colab compatibility
- Import statements checked for availability
- Syntax validated across all code cells

## Fixed

### Documentation
- Removed references to non-existent notebooks
- Fixed broken file paths
- Corrected outdated feature descriptions
- Updated notebook runtime estimates

### Testing
- Added missing notebook validation
- Improved CI workflow reliability
- Enhanced error reporting

## Deprecated

- None

## Removed

### Documentation
- References to `casp16_loader.py` (doesn't exist yet)
- Broken links to missing notebooks
- Outdated feature claims

## Security

### Added
- Credential exposure detection in notebooks
- Hardcoded path scanning
- Security-focused code review automation

---

## [Previous Versions]

### v0.2.0 - Production Training Pipelines
- Added A100 ULTIMATE MAXIMIZED notebook (200M parameters)
- Added A100 production training notebook (85M parameters)
- Implemented FAPE loss and IPA architecture
- ESM-2 and ProtT5 embedding integration

### v0.1.0 - Initial Release
- Basic quantum-classical hybrid model
- Quantum layer implementations
- Simple training pipeline
- Example notebooks

---

## Notes

### Versioning Strategy
- **Major version** (X.0.0): Breaking API changes, major architecture revisions
- **Minor version** (0.X.0): New features, notebooks, infrastructure (backward compatible)
- **Patch version** (0.0.X): Bug fixes, documentation updates, minor improvements

### Contribution Guidelines
See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- Submitting changes
- Coding standards
- Testing requirements
- Documentation expectations

### Release Process
1. Update CHANGELOG.md with all changes
2. Run full test suite (`pytest tests/`)
3. Verify all notebooks execute successfully
4. Update version in `pyproject.toml`
5. Create git tag (`git tag -a vX.Y.Z -m "Release vX.Y.Z"`)
6. Push to GitHub (`git push origin vX.Y.Z`)
7. Create GitHub release with notes from CHANGELOG

---

**Last Updated:** February 1, 2026
