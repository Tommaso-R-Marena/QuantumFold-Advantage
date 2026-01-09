# Contributing to QuantumFold-Advantage

Thank you for your interest in contributing to QuantumFold-Advantage! This document provides guidelines and instructions for contributing to the project.

## 🎯 How to Contribute

There are many ways to contribute:

1. **Report bugs** - Submit detailed bug reports via GitHub Issues
2. **Suggest features** - Propose new features or improvements
3. **Improve documentation** - Fix typos, clarify explanations, add examples
4. **Write code** - Fix bugs, implement features, optimize performance
5. **Add tests** - Improve test coverage
6. **Share results** - Contribute benchmark results or case studies

## 📋 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/QuantumFold-Advantage.git
cd QuantumFold-Advantage

# Add upstream remote
git remote add upstream https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .[dev,protein-lm,tracking]

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Always create a new branch for your changes
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## 💻 Development Workflow

### Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatting**: Use `black` for automatic formatting
- **Import sorting**: Use `isort` with black-compatible settings
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Follow Google style docstrings

```python
def my_function(param1: str, param2: int = 5) -> Dict[str, Any]:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 5)
    
    Returns:
        Dictionary containing results
    
    Raises:
        ValueError: If param1 is empty
    """
    pass
```

### Running Code Quality Checks

```bash
# Format code with black
black src tests

# Sort imports
isort src tests

# Check for style issues
flake8 src tests

# Type checking
mypy src

# Run all checks
pre-commit run --all-files
```

### Testing

All code contributions should include tests.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_quantum_layers.py

# Run tests matching pattern
pytest -k "test_esm"

# Run fast tests only (skip slow integration tests)
pytest -m "not slow"
```

### Test Structure

```python
import pytest
import torch
from src.quantum_layers import QuantumLayer


class TestQuantumLayer:
    """Tests for QuantumLayer class."""
    
    def test_initialization(self):
        """Test that QuantumLayer initializes correctly."""
        layer = QuantumLayer(n_qubits=4, depth=2)
        assert layer.n_qubits == 4
        assert layer.depth == 2
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        layer = QuantumLayer(n_qubits=4, depth=2)
        x = torch.randn(2, 10, 16)  # (batch, seq, features)
        output = layer(x)
        assert output.shape == (2, 10, 16)
    
    @pytest.mark.slow
    def test_training_convergence(self):
        """Test that layer can be trained (slow test)."""
        # Training test here
        pass
```

## 📝 Commit Guidelines

### Commit Message Format

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**

```
feat: Add CASP16 data loader with caching

Implements automatic download and caching of CASP16 targets.
Includes validation and error handling for missing structures.

Closes #42
```

```
fix: Resolve ESM2Embedder KeyError with -1 layer index

Automatically converts -1 to num_layers for proper indexing.
Adds validation to catch invalid layer indices.

Fixes #38
```

### Commit Best Practices

- Make atomic commits (one logical change per commit)
- Write meaningful commit messages
- Reference issue numbers when applicable
- Keep commits focused and small

## 🔬 Adding New Features

### 1. Plan the Feature

- Open an issue to discuss the feature
- Get feedback from maintainers
- Design the API interface

### 2. Implement

- Write the code following style guidelines
- Add comprehensive docstrings
- Include type hints
- Handle edge cases and errors

### 3. Test

- Write unit tests for all functions
- Add integration tests if applicable
- Test edge cases and error conditions
- Ensure tests pass locally

### 4. Document

- Update relevant documentation
- Add docstrings to all public functions
- Update README if needed
- Add examples if appropriate

## 🐛 Fixing Bugs

### 1. Reproduce

- Create a minimal reproducible example
- Document the expected vs. actual behavior
- Note your environment (OS, Python version, package versions)

### 2. Fix

- Write a failing test that demonstrates the bug
- Fix the bug
- Verify the test now passes
- Check for similar bugs elsewhere

### 3. Test

- Run the full test suite
- Test edge cases
- Verify the fix doesn't break other functionality

## 📊 Performance Optimization

When optimizing performance:

1. **Profile first** - Use `cProfile` or `line_profiler` to identify bottlenecks
2. **Measure impact** - Benchmark before and after
3. **Document tradeoffs** - Explain any readability vs. performance tradeoffs
4. **Add benchmarks** - Include performance tests to prevent regressions

```python
import pytest
import time

@pytest.mark.benchmark
def test_quantum_layer_performance():
    """Benchmark quantum layer forward pass."""
    layer = QuantumLayer(n_qubits=6)
    x = torch.randn(32, 100, 64)
    
    start = time.time()
    for _ in range(10):
        _ = layer(x)
    elapsed = time.time() - start
    
    # Should complete 10 iterations in under 5 seconds
    assert elapsed < 5.0
```

## 📚 Documentation

### Docstring Style

Use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: List[int],
    param3: Optional[Dict] = None,
) -> Tuple[bool, str]:
    """Brief one-line description.
    
    More detailed description can go here. Explain the purpose,
    key concepts, and any important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter. Can span
            multiple lines if needed.
        param3: Optional parameter description (default: None)
    
    Returns:
        Tuple containing:
            - bool: Success status
            - str: Result message or error description
    
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 contains non-integers
    
    Example:
        >>> result, message = complex_function("test", [1, 2, 3])
        >>> print(result)
        True
    
    Note:
        Important notes about usage, limitations, or gotchas.
    """
    pass
```

### README Updates

When adding significant features:

1. Update the relevant section of README.md
2. Add usage examples
3. Update the feature list
4. Update installation instructions if needed

## 🔍 Code Review Process

### Submitting a Pull Request

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**
   ```bash
   pytest
   black src tests
   isort src tests
   flake8 src tests
   mypy src
   ```

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature
   ```

4. **Create Pull Request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what changed and why
   - Include any breaking changes
   - Add screenshots if UI-related

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Closes #XX

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Updated documentation

## Screenshots (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex sections
- [ ] Updated documentation
- [ ] No new warnings
- [ ] Added tests
- [ ] All tests pass
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, maintainers will merge

## 🎓 Learning Resources

### Understanding the Codebase

- **Quantum Computing**: Start with `src/quantum_layers.py`
- **Deep Learning**: Check `src/advanced_model.py`
- **Protein Embeddings**: See `src/protein_embeddings.py`
- **Training**: Review `src/advanced_training.py`

### Recommended Reading

- [PennyLane Documentation](https://pennylane.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [AlphaFold2 Paper](https://www.nature.com/articles/s41586-021-03819-2)
- [ESM-2 Paper](https://www.science.org/doi/10.1126/science.ade2574)

## 💬 Communication

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: marena@cua.edu (for sensitive issues only)

### Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Give credit where due
- Focus on the code, not the person

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in the contributors section
- Acknowledged in release notes
- Mentioned in publications (for significant contributions)

---

**Thank you for contributing to QuantumFold-Advantage!** 🚀
