# Multi-stage build for faster CI/CD
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies (will be removed in final stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only dependency files first for better caching
COPY requirements.txt pyproject.toml ./

# Install core dependencies with pre-built wheels (FAST)
# Split into layers for better caching
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    'numpy>=1.24.0,<2.0.0' \
    'scipy>=1.10.0' \
    'pandas>=2.0.0'

RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    'torch>=2.0.0' \
    'torchvision' \
    'torchaudio'

RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    'autoray>=0.6.11' \
    'pennylane>=0.33.0,<0.42.0' \
    'pennylane-lightning>=0.33.0,<0.39.0'

# Install biotite and biopython (these can be slow)
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    'biotite>=0.38.0,<1.0.0' \
    'biopython>=1.81,<2.0.0'

# Install remaining dependencies
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    'matplotlib>=3.7.0' \
    'seaborn>=0.12.0' \
    'scikit-learn>=1.3.0' \
    'tqdm>=4.66.0' \
    'pyyaml>=6.0' \
    'h5py>=3.9.0' \
    'joblib>=1.3.0'

# Copy application code
COPY . .

# Build the package wheel
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# ============================================
# Final stage - minimal image
# ============================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder
COPY --from=builder /wheels /wheels

# Install all wheels at once (FAST - no compilation)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links /wheels /wheels/quantumfold_advantage*.whl && \
    rm -rf /wheels ~/.cache/pip

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 quantumfold && \
    chown -R quantumfold:quantumfold /app

USER quantumfold

# HTTP port
EXPOSE 8000

# Jupyter port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import src; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.cli"]