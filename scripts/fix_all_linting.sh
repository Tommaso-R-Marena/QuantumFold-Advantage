#!/bin/bash
# Comprehensive fix script for all linting errors

set -e

echo "========================================"
echo "COMPREHENSIVE LINTING FIX SCRIPT"
echo "========================================"
echo ""

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from repository root"
    exit 1
fi

# Install required tools
echo "[1/5] Installing linting tools..."
pip install -q autoflake isort black flake8

# Remove unused imports and variables
echo "[2/5] Removing unused imports and variables..."
autoflake \
    --remove-all-unused-imports \
    --remove-unused-variables \
    --ignore-init-module-imports \
    --in-place \
    --recursive \
    src/ tests/ \
    --exclude "__pycache__,*.pyc,.pytest_cache"

echo "   ✅ Unused imports removed"

# Sort imports
echo "[3/5] Sorting imports..."
isort src/ tests/ --profile black --line-length 100
echo "   ✅ Imports sorted"

# Format code with Black
echo "[4/5] Formatting code with Black..."
black src/ tests/ --line-length 100
echo "   ✅ Code formatted"

# Run flake8 to check
echo "[5/5] Running flake8 check..."
if flake8 src/ tests/ --count --statistics; then
    echo ""
    echo "========================================"
    echo "✅ ALL LINTING FIXES COMPLETE!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review changes: git diff"
    echo "  2. Test locally: pytest tests/"
    echo "  3. Commit: git add -A && git commit -m 'fix: Apply comprehensive linting fixes'"
    echo "  4. Push: git push"
else
    echo ""
    echo "⚠️  Some linting issues remain (see above)"
    echo "These may require manual fixes"
    exit 1
fi
