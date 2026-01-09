# QuantumFold-Advantage Docker Container
# Reproducible environment for quantum-classical protein structure prediction
#
# Build: docker build -t quantumfold-advantage .
# Run: docker run --gpus all -p 8888:8888 -v $(pwd)/data:/workspace/data quantumfold-advantage

# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Metadata
LABEL maintainer="Tommaso R. Marena <marena@cua.edu>"
LABEL description="QuantumFold-Advantage: Quantum-Classical Hybrid Protein Folding"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24.0,<2.0.0 \
    scipy>=1.10.0,<2.0.0 \
    torch>=2.0.0,<2.3.0 --index-url https://download.pytorch.org/whl/cu118 \
    torchvision>=0.15.0,<0.18.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    autoray>=0.6.11 \
    pennylane>=0.33.0,<0.36.0 \
    pennylane-lightning>=0.33.0,<0.36.0

RUN pip install --no-cache-dir \
    fair-esm>=2.0.0 \
    transformers>=4.35.0,<5.0.0 \
    sentencepiece>=0.1.99,<1.0.0 \
    tokenizers>=0.15.0,<1.0.0 \
    huggingface-hub>=0.19.0

RUN pip install --no-cache-dir \
    biopython>=1.81,<2.0.0 \
    biotite>=0.38.0,<1.0.0 \
    pandas>=2.0.0,<3.0.0 \
    h5py>=3.9.0,<4.0.0 \
    pyyaml>=6.0,<7.0.0 \
    joblib>=1.3.0

RUN pip install --no-cache-dir \
    matplotlib>=3.7.0,<4.0.0 \
    seaborn>=0.12.0,<1.0.0 \
    plotly>=5.17.0,<6.0.0 \
    kaleido>=0.2.1

RUN pip install --no-cache-dir \
    scikit-learn>=1.3.0,<2.0.0 \
    statsmodels>=0.14.0,<1.0.0 \
    tqdm>=4.66.0 \
    requests>=2.31.0,<3.0.0 \
    pillow>=10.0.0,<11.0.0

RUN pip install --no-cache-dir \
    jupyterlab>=4.0.0,<5.0.0 \
    ipywidgets>=8.1.0,<9.0.0 \
    ipykernel>=6.25.0,<7.0.0 \
    tensorboard>=2.14.0,<3.0.0 \
    pytest>=7.4.0,<9.0.0

# Copy source code
COPY . /workspace/

# Install QuantumFold-Advantage in editable mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/outputs /workspace/checkpoints /workspace/logs

# JupyterLab port
EXPOSE 8888
# TensorBoard port
EXPOSE 6006
# API port (if using FastAPI)
EXPOSE 8000

# Set up JupyterLab configuration
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "QuantumFold-Advantage Docker Container"\n\
echo "======================================="\n\
echo ""\n\
echo "Available commands:"\n\
echo "  jupyter    - Start JupyterLab server"\n\
echo "  tensorboard - Start TensorBoard"\n\
echo "  train      - Run training script"\n\
echo "  test       - Run pytest"\n\
echo "  bash       - Interactive bash shell"\n\
echo ""\n\
\n\
case "$1" in\n\
    jupyter)\n\
        echo "Starting JupyterLab on port 8888..."\n\
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n\
        ;;\n\
    tensorboard)\n\
        echo "Starting TensorBoard on port 6006..."\n\
        tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006\n\
        ;;\n\
    train)\n\
        shift\n\
        echo "Running training..."\n\
        python train_advanced.py "$@"\n\
        ;;\n\
    test)\n\
        echo "Running tests..."\n\
        pytest tests/ -v\n\
        ;;\n\
    bash)\n\
        /bin/bash\n\
        ;;\n\
    *)\n\
        echo "Starting JupyterLab by default..."\n\
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n\
        ;;\n\
esac' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/ || exit 1
