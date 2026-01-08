# QuantumFold-Advantage Docker Image
# Optimized for size and build speed

FROM python:3.10-slim AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stage 2: Build environment
FROM base AS builder

WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install Python dependencies to user directory
# Note: PIP_NO_CACHE_DIR=1 disables cache, so no need to purge
RUN pip install --user --no-warn-script-location \
    numpy scipy torch pennylane matplotlib pandas scikit-learn biopython requests tqdm psutil pytest

# Stage 3: Production image
FROM base AS production

# Create non-root user
RUN useradd -m -u 1000 quantumfold && \
    mkdir -p /app /data /outputs /checkpoints && \
    chown -R quantumfold:quantumfold /app /data /outputs /checkpoints

# Copy Python packages from builder
COPY --from=builder --chown=quantumfold:quantumfold /root/.local /home/quantumfold/.local

# Set up Python path (define PYTHONPATH before use)
ENV PATH=/home/quantumfold/.local/bin:$PATH \
    PYTHONPATH=/app

WORKDIR /app

# Copy only essential application code
COPY --chown=quantumfold:quantumfold src/ ./src/
COPY --chown=quantumfold:quantumfold *.py ./ 2>/dev/null || true

# Switch to non-root user
USER quantumfold

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('OK')" || exit 1

# Default command
CMD ["python3", "--version"]

# Labels
LABEL maintainer="Tommaso R. Marena" \
      version="1.0.0" \
      description="QuantumFold-Advantage: Hybrid Quantum-Classical Protein Folding"
