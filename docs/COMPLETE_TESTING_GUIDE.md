# Complete Testing Guide - QuantumFold-Advantage

**Last Updated:** January 12, 2026  
**Status:** ✅ Production-Ready  
**Total Tests:** 150+  
**Test Coverage:** 100% (notebooks)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Test Infrastructure](#test-infrastructure)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Notebook Testing](#notebook-testing)
6. [CI/CD Integration](#cicd-integration)
7. [Performance Benchmarking](#performance-benchmarking)
8. [Contributing Tests](#contributing-tests)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

### Test Suite Statistics

| Category | Files | Tests | Purpose |
|----------|-------|-------|----------|
| **Notebook Component Tests** | 4 | 143 | Test code from notebooks |
| **Notebook Execution Tests** | 1 | 15 | End-to-end notebook runs |
| **Unit Tests** | 12 | 50+ | Core functionality |
| **Integration Tests** | 3 | 20+ | Full pipeline testing |
| **Performance Tests** | 2 | 10+ | Benchmarking |
| **Total** | **22** | **150+** | **Complete coverage** |

### Quality Metrics

✅ **100% notebook coverage** - All 5 example notebooks tested  
✅ **143 notebook component tests** - Every major function validated  
✅ **End-to-end execution** - Notebooks run without errors  
✅ **CI/CD integrated** - Automatic testing on every push  
✅ **Performance benchmarked** - Speed and memory tracked  

---

## 🏗️ Test Infrastructure

### Directory Structure

```
tests/
├── conftest.py                      # Pytest configuration & fixtures
│
├── Notebook Component Tests (143 tests)
├── test_colab_quickstart.py         # 25 tests - Quick start notebook
├── test_01_getting_started.py       # 35 tests - Getting started
├── test_02_quantum_classical.py     # 46 tests - Training comparison
├── test_03_visualization.py         # 37 tests - Visualization
│
├── Notebook Execution Tests (15 tests)
├── test_notebook_execution.py       # Execute notebooks end-to-end
│
├── Core Unit Tests (50+ tests)
├── test_model.py                    # Model architecture
├── test_quantum_layers.py           # Quantum components
├── test_embeddings.py               # Protein embeddings
├── test_training.py                 # Training loops
├── test_data.py                     # Data processing
│
├── Integration Tests (20+ tests)
├── test_integration.py              # Full pipeline
├── test_benchmarks.py               # Benchmark tools
│
├── Advanced Tests (10+ tests)
├── test_performance.py              # Speed benchmarks
├── test_stress.py                   # Load testing
├── test_regression.py               # Regression detection
├── test_property.py                 # Property-based tests
│
└── README.md                        # Test documentation
```

### Key Test Files

#### **1. `test_colab_quickstart.py`** (25 tests)

Tests all functionality from `colab_quickstart.ipynb`:

- ✅ Environment setup and imports
- ✅ Model initialization and forward passes
- ✅ RMSD and TM-score calculations
- ✅ 3D visualization generation
- ✅ Training stability

```bash
pytest tests/test_colab_quickstart.py -v
```

#### **2. `test_01_getting_started.py`** (35 tests)

Tests advanced tutorial notebook:

- ✅ Path operations and imports
- ✅ SimpleProteinModel fallback
- ✅ CASP metrics (RMSD, TM-score, GDT_TS)
- ✅ Confidence scores (pLDDT)
- ✅ Distance map calculations

```bash
pytest tests/test_01_getting_started.py -v
```

#### **3. `test_02_quantum_classical.py`** (46 tests)

Tests training comparison notebook:

- ✅ Synthetic data generation
- ✅ DataLoader batching
- ✅ Training loop convergence
- ✅ Model comparison metrics
- ✅ Performance tracking
- ✅ Edge cases (empty data, NaN inputs)

```bash
pytest tests/test_02_quantum_classical.py -v
```

#### **4. `test_03_visualization.py`** (37 tests)

Tests visualization notebook:

- ✅ 3D plotting (matplotlib)
- ✅ Distance and contact maps
- ✅ Confidence heatmaps
- ✅ Ramachandran plots
- ✅ Plotly integration
- ✅ Figure saving

```bash
pytest tests/test_03_visualization.py -v
```

#### **5. `test_notebook_execution.py`** (15 tests)

**Most important for ensuring notebooks run!**

Actually executes notebooks end-to-end:

- ✅ Executes all cells in order
- ✅ Verifies outputs are generated
- ✅ Checks for errors
- ✅ Validates notebook structure
- ✅ Confirms imports work

```bash
# Run all execution tests (slow, ~10-30 min)
pytest tests/test_notebook_execution.py -v

# Run structure tests only (fast)
pytest tests/test_notebook_execution.py::TestNotebookStructure -v
```

---

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_colab_quickstart.py

# Run specific test
pytest tests/test_colab_quickstart.py::TestEnvironmentSetup::test_imports -v
```

### Test Selection

```bash
# Run only fast tests (skip notebook execution)
pytest -m "not slow"

# Run only slow tests
pytest -m slow

# Run only performance benchmarks
pytest -m performance

# Run tests matching pattern
pytest -k "visualization"

# Run tests for specific notebook
pytest -k "colab_quickstart"
```

### Coverage Reports

```bash
# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
pytest -n auto

# Specify number of workers
pytest -n 4
```

---

## 📊 Test Categories

### 1. Unit Tests

**Purpose:** Test individual functions and classes in isolation

```bash
pytest tests/test_model.py tests/test_quantum_layers.py
```

**Coverage:**
- Model initialization
- Forward passes
- Parameter counts
- Device placement
- Edge cases

### 2. Integration Tests

**Purpose:** Test complete workflows

```bash
pytest tests/test_integration.py
```

**Coverage:**
- End-to-end training
- Data loading → model → evaluation
- Multi-component interactions

### 3. Notebook Component Tests

**Purpose:** Test code snippets from notebooks

```bash
pytest tests/test_colab_quickstart.py tests/test_01_getting_started.py \
       tests/test_02_quantum_classical.py tests/test_03_visualization.py
```

**Coverage:**
- All functions used in notebooks
- Expected outputs
- Error handling

### 4. Notebook Execution Tests

**Purpose:** Ensure notebooks run end-to-end

```bash
pytest tests/test_notebook_execution.py -v
```

**Coverage:**
- Full notebook execution
- Cell-by-cell validation
- Output generation
- Error detection

### 5. Performance Tests

**Purpose:** Benchmark speed and memory

```bash
pytest tests/test_performance.py -v
```

**Coverage:**
- Forward pass speed
- Memory usage
- Scaling with sequence length
- GPU utilization

---

## 📓 Notebook Testing

### Component Testing

**Tests individual code blocks from notebooks without executing the entire notebook.**

✅ **Advantages:**
- Fast (< 1 minute)
- Isolates failures
- Runs in CI easily

❌ **Limitations:**
- Doesn't test full workflow
- May miss integration issues

### Execution Testing

**Actually runs the notebook files end-to-end.**

✅ **Advantages:**
- Guarantees notebooks work
- Tests real user experience
- Catches integration issues

❌ **Limitations:**
- Slow (2-30 minutes per notebook)
- Requires more resources
- Can timeout in CI

### Best Practice: Use Both!

```bash
# Fast feedback during development
pytest tests/test_colab_quickstart.py

# Before committing changes
pytest tests/test_notebook_execution.py::TestNotebookExecution::test_colab_quickstart_executes
```

---

## 🔄 CI/CD Integration

### GitHub Actions Workflows

#### **1. Main CI Workflow** (`.github/workflows/ci.yml`)

Runs on every push and PR:

- ✅ All unit tests
- ✅ Integration tests
- ✅ Notebook component tests
- ⏱️ ~5-10 minutes

#### **2. Notebook Execution Workflow** (`.github/workflows/test-notebooks-execution.yml`)

Runs on notebook changes:

- ✅ Executes all notebooks
- ✅ Validates outputs
- ✅ Uploads executed notebooks as artifacts
- ⏱️ ~20-30 minutes

#### **3. Status** Checks (`.github/workflows/status-checks.yml`)

Combined status for required checks

### Triggering CI

```bash
# Push triggers CI
git push origin main

# PR triggers CI
gh pr create

# Manual trigger
gh workflow run test-notebooks-execution.yml
```

### Viewing Results

1. Go to [Actions tab](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions)
2. Click on workflow run
3. View logs and artifacts
4. Download executed notebooks if needed

---

## ⚡ Performance Benchmarking

### Running Benchmarks

```bash
# Run all performance tests
pytest tests/test_performance.py -v

# Run with pytest-benchmark (detailed stats)
pytest tests/test_performance.py --benchmark-only

# Save benchmark results
pytest tests/test_performance.py --benchmark-save=baseline

# Compare against baseline
pytest tests/test_performance.py --benchmark-compare=baseline
```

### Benchmark Output

```
test_forward_pass_speed
  Mean: 45.2ms ± 2.1ms
  Min: 42.8ms
  Max: 51.3ms
  
test_memory_usage
  Peak: 128.4 MB
  
test_scaling_with_sequence_length
  Seq 10:  0.0421s
  Seq 20:  0.0893s
  Seq 50:  0.3201s
  Seq 100: 1.2456s
```

### Performance Regression Testing

```bash
# Set baseline
pytest tests/test_performance.py --benchmark-save=v1.0

# After changes, compare
pytest tests/test_performance.py --benchmark-compare=v1.0

# Fail if regression > 10%
pytest tests/test_performance.py --benchmark-compare=v1.0 --benchmark-max-deviation=10
```

---

## 🧪 Contributing Tests

### When to Add Tests

✅ **Always add tests for:**
- New features
- Bug fixes
- Performance-critical code
- Public API changes

### Test Structure

```python
import pytest
import torch
from src.my_module import MyClass


class TestMyClass:
    """Tests for MyClass."""
    
    def test_initialization(self):
        """Test that MyClass initializes correctly."""
        obj = MyClass(param=42)
        assert obj.param == 42
    
    def test_main_functionality(self):
        """Test main use case."""
        obj = MyClass(param=42)
        result = obj.process(torch.randn(10))
        assert result.shape == (10,)
    
    def test_edge_case_empty_input(self):
        """Test edge case: empty input."""
        obj = MyClass(param=42)
        with pytest.raises(ValueError):
            obj.process(torch.tensor([]))
    
    @pytest.mark.slow
    def test_performance(self):
        """Test performance is acceptable (slow test)."""
        obj = MyClass(param=42)
        import time
        start = time.time()
        obj.process(torch.randn(1000, 1000))
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should complete in < 1 second
```

### Running Pre-commit Checks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 🔧 Troubleshooting

### Common Issues

#### **1. NumPy 2.0 Incompatibility**

**Symptoms:**
```
AttributeError: numpy.ndarray' object has no attribute 'item_size'
```

**Solution:**
```bash
pip uninstall -y numpy jax jaxlib autograd pennylane
pip install --force-reinstall --no-deps 'numpy>=1.23.0,<2.0.0'
pip install --no-deps 'autograd>=1.6.2'
pip install --no-deps 'pennylane>=0.32.0'
```

#### **2. Notebook Execution Timeout**

**Symptoms:**
```
CellExecutionError: timeout
```

**Solution:**
```bash
# Increase timeout
pytest tests/test_notebook_execution.py --timeout=1200

# Or skip slow notebooks
pytest tests/test_notebook_execution.py -m "not slow"
```

#### **3. CUDA Out of Memory**

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size in tests
# Or run on CPU
pytest tests/ --device=cpu
```

#### **4. Import Errors**

**Symptoms:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Getting Help

1. **Check documentation:** [README](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/blob/main/README.md)
2. **Search issues:** [GitHub Issues](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/issues)
3. **Ask for help:** [Open new issue](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/issues/new)

---

## 📚 Additional Resources

### Documentation

- **[README](../README.md)** - Project overview
- **[CONTRIBUTING](../CONTRIBUTING.md)** - Contribution guidelines
- **[NOTEBOOK_ANALYSIS](./NOTEBOOK_ANALYSIS.md)** - Detailed notebook analysis

### External Resources

- **[Pytest Documentation](https://docs.pytest.org/)** - Testing framework
- **[nbconvert Documentation](https://nbconvert.readthedocs.io/)** - Notebook execution
- **[GitHub Actions](https://docs.github.com/en/actions)** - CI/CD

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Run all tests
pytest

# Run fast tests only
pytest -m "not slow"

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_colab_quickstart.py

# Run notebook execution tests
pytest tests/test_notebook_execution.py

# Run performance benchmarks
pytest tests/test_performance.py

# Run in parallel
pytest -n auto
```

### Test Markers

- `@pytest.mark.slow` - Slow tests (> 10 seconds)
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.integration` - Integration tests

---

**Last Updated:** January 12, 2026  
**Maintainer:** Tommaso R. Marena  
**Status:** ✅ **150+ tests** • **100% notebook coverage** • **CI-integrated**