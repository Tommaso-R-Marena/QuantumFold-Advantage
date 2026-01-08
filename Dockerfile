# QuantumFold-Advantage Docker Image
# Multi-stage build for optimized image size

# Stage 1: Base image with dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stage 2: Build environment
FROM base AS builder

WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage 3: Production image
FROM base AS production

# Create non-root user
RUN useradd -m -u 1000 quantumfold && \
    mkdir -p /app /data /outputs /checkpoints && \
    chown -R quantumfold:quantumfold /app /data /outputs /checkpoints

# Copy Python packages from builder
COPY --from=builder --chown=quantumfold:quantumfold /root/.local /home/quantumfold/.local

# Set up Python path
ENV PATH=/home/quantumfold/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH

WORKDIR /app

# Copy application code
COPY --chown=quantumfold:quantumfold . .

# Switch to non-root user
USER quantumfold

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import pennylane; print('OK')" || exit 1

# Default command
CMD ["python3", "run_demo.py"]

# Labels
LABEL maintainer="Tommaso R. Marena" \
      version="1.0.0" \
      description="QuantumFold-Advantage: Hybrid Quantum-Classical Protein Folding"
