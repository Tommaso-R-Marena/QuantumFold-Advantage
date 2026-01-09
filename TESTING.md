# Testing Guide - QuantumFold-Advantage

Comprehensive testing documentation for developers and contributors.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Test Markers](#test-markers)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Writing Tests](#writing-tests)
7. [Coverage Requirements](#coverage-requirements)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Install Test Dependencies

```bash
# Full development setup
pip install -e .[dev]

# Or just testing tools
pip install pytest pytest-cov pytest-xdist pytest-timeout
```

### Run All Tests

```bash
# Fast tests only (skip slow/GPU tests)
pytest tests/ -v -m "not slow and not gpu"

# All tests including slow ones
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/ -m "not integration and not slow"

# Integration tests
pytest tests/ -m integration

# Performance tests
pytest tests/ -m performance

# Stress tests
pytest tests/ -m stress
```

---

## Test Categories

### 1. Unit Tests
**Purpose:** Test individual components in isolation  
**Location:** `tests/test_*.py`  
**Speed:** Fast (<1s per test)  
**Coverage:** Core functionality

### 2. Integration Tests
**Purpose:** Test component interactions  
**Location:** `tests/test_integration.py`  
**Speed:** Medium (1-10s per test)  
**Coverage:** Full pipeline workflows

### 3. Performance Tests
**Purpose:** Benchmark speed and resource usage  
**Location:** `tests/test_performance.py`  
**Speed:** Medium  
**Tools:** pytest-benchmark, memory_profiler

### 4. Stress Tests
**Purpose:** Test edge cases and limits  
**Location:** `tests/test_stress.py`  
**Speed:** Slow (10s-1min per test)  
**Coverage:** Edge cases, large inputs

### 5. Property-Based Tests
**Purpose:** Test invariants with random inputs  
**Location:** `tests/test_property.py`  
**Speed:** Medium  
**Tools:** Hypothesis

### 6. Regression Tests
**Purpose:** Catch breaking changes  
**Location:** `tests/test_regression.py`  
**Speed:** Fast  
**Coverage:** API stability

---

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestAdvancedProteinFoldingModel::test_forward_pass
```

### Parallel Execution

```bash
# Run tests in parallel (4 workers)
pytest -n 4

# Auto-detect number of CPUs
pytest -n auto
```

### With Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing

# HTML coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html

# XML for CI
pytest --cov=src --cov-report=xml
```

### Filtering Tests

```bash
# By marker
pytest -m slow              # Only slow tests
pytest -m "not slow"        # Skip slow tests
pytest -m "integration or performance"

# By test name pattern
pytest -k "test_model"      # Tests with 'model' in name
pytest -k "not quantum"     # Skip tests with 'quantum' in name
```

---

## Test Markers

All available test markers:

### Core Markers

- `@pytest.mark.slow` - Tests taking >10 seconds
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.stress` - Stress/edge case tests
- `@pytest.mark.property` - Property-based tests
- `@pytest.mark.regression` - Regression tests

### Usage in Tests

```python
import pytest

@pytest.mark.slow
def test_long_running():
    # This test takes a while
    pass

@pytest.mark.gpu
def test_cuda_operations():
    # Requires GPU
    pass

@pytest.mark.integration
def test_full_pipeline():
    # Tests multiple components together
    pass
```

---

## CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Main CI (`.github/workflows/ci.yml`)

**Triggers:** Every push and PR  
**Matrix:**
- Python: 3.8, 3.9, 3.10, 3.11
- OS: Ubuntu, macOS, Windows
- Tests: Fast unit tests only

**Checks:**
- Code quality (Black, isort, flake8)
- Unit tests with coverage
- Docker build
- Documentation

#### 2. Comprehensive CI (`.github/workflows/ci-comprehensive.yml`)

**Triggers:** Push to main, PRs, daily schedule  
**Tests:**
- All unit tests
- Integration tests
- Slow tests
- Performance tests
- Dependency security scan

#### 3. Advanced Testing (`.github/workflows/advanced-testing.yml`)

**Triggers:** Weekly, on-demand  
**Tests:**
- Mutation testing
- Property-based testing
- Stress tests
- Notebook validation
- Memory leak detection
- Regression tests

### Status Badges

Add to README:

```markdown
[![CI](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/ci.yml/badge.svg)](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Tommaso-R-Marena/QuantumFold-Advantage/branch/main/graph/badge.svg)](https://codecov.io/gh/Tommaso-R-Marena/QuantumFold-Advantage)
```

---

## Writing Tests

### Test Structure

```python
"""Module docstring describing test file."""

import pytest
import torch

try:
    from src.my_module import MyClass
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False


@pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
class TestMyClass:
    """Test suite for MyClass."""
    
    def test_initialization(self):
        """Test object initialization."""
        obj = MyClass()
        assert obj is not None
    
    def test_forward_pass(self, sample_data):
        """Test forward pass with fixtures."""
        obj = MyClass()
        output = obj(sample_data)
        assert output.shape == sample_data.shape
    
    @pytest.mark.slow
    def test_training_loop(self):
        """Test full training (slow)."""
        # Training code here
        pass
```

### Using Fixtures

```python
# Use built-in fixtures from conftest.py
def test_with_fixtures(
    sample_embeddings,
    sample_coordinates,
    temp_output_dir,
    device
):
    model = Model(device=device)
    output = model(sample_embeddings)
    # Save to temp_output_dir
```

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<what_it_tests>`
- Be descriptive: `test_forward_pass_with_batch_norm`

---

## Coverage Requirements

### Targets

- **Minimum:** 70% overall coverage
- **Goal:** 80%+ coverage
- **Critical modules:** 90%+ coverage

### Checking Coverage

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# See which lines are not covered
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=70
```

### Coverage Configuration

In `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

---

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Make sure package is installed in editable mode
pip install -e .

# Check PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### PennyLane/Autoray Issues

```bash
# Ensure correct autoray version
pip install "autoray>=0.6.11"
pip install pennylane pennylane-lightning
```

#### GPU Tests Failing

```bash
# Skip GPU tests
pytest -m "not gpu"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Slow Tests Taking Too Long

```bash
# Skip slow tests
pytest -m "not slow"

# Set timeout
pytest --timeout=60  # 60 seconds max per test
```

### Getting Help

1. Check test logs: `pytest -v --tb=long`
2. Run single failing test: `pytest tests/test_file.py::test_name -vv`
3. Check fixtures: `pytest --fixtures`
4. Debug mode: `pytest --pdb` (drops into debugger on failure)

---

## Best Practices

### DO

✅ Write tests for new features  
✅ Use fixtures for shared setup  
✅ Mark slow tests with `@pytest.mark.slow`  
✅ Test edge cases  
✅ Keep tests independent  
✅ Use descriptive test names  
✅ Add docstrings to test functions  

### DON'T

❌ Don't test implementation details  
❌ Don't use global state  
❌ Don't write flaky tests  
❌ Don't ignore failing tests  
❌ Don't hardcode file paths  
❌ Don't skip writing tests  

---

## Local Pre-commit Checks

### Install Pre-commit

```bash
pip install pre-commit
pre-commit install
```

### Run Manually

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### Before Pushing

```bash
# Quick check
black src tests && isort src tests && pytest -m "not slow"

# Full check (like CI)
black --check src tests && \
isort --check src tests && \
flake8 src tests && \
pytest --cov=src --cov-fail-under=70
```

---

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)

---

**Questions?** Open an issue or check [CONTRIBUTING.md](CONTRIBUTING.md)
