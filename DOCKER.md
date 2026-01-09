# Docker Deployment Guide

This guide explains how to run QuantumFold-Advantage in Docker containers with GPU support.

## Prerequisites

### Required
- [Docker](https://docs.docker.com/get-docker/) >= 20.10
- [Docker Compose](https://docs.docker.com/compose/install/) >= 2.0
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Verify GPU Access

```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU information displayed.

## Quick Start

### 1. Build the Image

```bash
# Clone repository
git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git
cd QuantumFold-Advantage

# Build image
docker-compose build
```

### 2. Start Jupyter Lab

```bash
# Start container with Jupyter
docker-compose up

# Access Jupyter at: http://localhost:8888
```

### 3. Run Training

```bash
# Start training service
docker-compose --profile training up training
```

### 4. Start API Server

```bash
# Start API service
docker-compose --profile api up api

# Access API at: http://localhost:8000
# Docs at: http://localhost:8000/docs
```

## Usage Examples

### Interactive Shell

```bash
# Start container with bash
docker-compose run --rm quantumfold /bin/bash

# Inside container:
python train_advanced.py --help
```

### Run Training Script

```bash
# Quantum model training
docker-compose run --rm quantumfold python train_advanced.py \
    --use-quantum \
    --use-amp \
    --use-ema \
    --epochs 100 \
    --batch-size 32 \
    --output-dir /workspace/outputs/quantum_run

# Classical baseline
docker-compose run --rm quantumfold python train_advanced.py \
    --epochs 100 \
    --batch-size 32 \
    --output-dir /workspace/outputs/classical_run
```

### Run Complete Benchmark

```bash
# Execute benchmark notebook
docker-compose run --rm quantumfold jupyter nbconvert \
    --to notebook \
    --execute examples/complete_benchmark.ipynb \
    --output /workspace/outputs/benchmark_results.ipynb
```

### TensorBoard Monitoring

```bash
# Start TensorBoard
docker-compose --profile tensorboard up tensorboard

# Access at: http://localhost:6006
```

## Volume Mounts

The docker-compose configuration mounts several directories:

```yaml
volumes:
  - ./data:/workspace/data          # Input data
  - ./outputs:/workspace/outputs    # Training outputs
  - ./models:/workspace/models      # Saved models
  - ./logs:/workspace/logs          # Training logs
  - ./src:/workspace/src            # Source code (for development)
  - ./configs:/workspace/configs    # Configuration files
```

### Adding Your Data

```bash
# Create data directory
mkdir -p data/proteins

# Copy your protein files
cp /path/to/your/proteins/*.pdb data/proteins/

# Files are now accessible in container at /workspace/data/
```

## GPU Configuration

### Select Specific GPU

```bash
# Use GPU 1 only
CUDA_VISIBLE_DEVICES=1 docker-compose up
```

### Multi-GPU Training

Modify `docker-compose.yml`:

```yaml
resources:
  reservations:
    devices:
      - driver: nvidia
        count: all  # Use all GPUs
        capabilities: [gpu]
```

## Memory Configuration

### Increase Shared Memory

For large batch sizes, increase shared memory:

```yaml
shm_size: '16gb'  # Default is 8gb
```

### Limit GPU Memory

```bash
# Limit to 8GB VRAM
docker-compose run --rm \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    quantumfold python train_advanced.py
```

## Development Mode

### Live Code Reloading

Source code is mounted as a volume, so changes are reflected immediately:

```bash
# Edit code on host
vim src/quantum_layers.py

# Restart Jupyter kernel or re-run script
# Changes are automatically available
```

### Install Additional Packages

```bash
# Interactive shell
docker-compose run --rm quantumfold /bin/bash

# Inside container
pip install your-package

# To persist, add to requirements.txt and rebuild
```

## Production Deployment

### Build Production Image

```bash
# Build with specific tag
docker build -t quantumfold-advantage:v1.0 .

# Push to registry
docker tag quantumfold-advantage:v1.0 your-registry/quantumfold-advantage:v1.0
docker push your-registry/quantumfold-advantage:v1.0
```

### Run Production Container

```bash
# Production training
docker run --rm --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    quantumfold-advantage:v1.0 \
    python train_advanced.py --config configs/advanced_config.yaml
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon
sudo systemctl status docker

# Reinstall NVIDIA Container Toolkit if needed
```

### Out of Memory

```bash
# Reduce batch size
docker-compose run --rm quantumfold python train_advanced.py --batch-size 8

# Use gradient accumulation
docker-compose run --rm quantumfold python train_advanced.py \
    --batch-size 8 \
    --gradient-accumulation-steps 4
```

### Permission Issues

```bash
# Fix volume permissions
sudo chown -R $(id -u):$(id -g) outputs/ logs/ models/
```

### Port Already in Use

```bash
# Use different port
docker-compose run --rm -p 8889:8888 quantumfold

# Or stop conflicting service
sudo lsof -i :8888
sudo kill -9 <PID>
```

## Cleanup

### Remove Containers

```bash
# Stop and remove containers
docker-compose down

# Remove with volumes
docker-compose down -v
```

### Remove Images

```bash
# Remove built image
docker rmi quantumfold-advantage:latest

# Remove all unused images
docker image prune -a
```

### Clean Everything

```bash
# Nuclear option: remove all Docker data
docker system prune -a --volumes
```

## Advanced Usage

### Custom Dockerfile

Create `Dockerfile.custom`:

```dockerfile
FROM quantumfold-advantage:latest

# Add your customizations
RUN pip install your-extra-packages

COPY your-scripts/ /workspace/scripts/
```

### Multi-Stage Build

For smaller production images, use multi-stage builds:

```dockerfile
# Build stage
FROM python:3.10 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
```

### Orchestration with Kubernetes

See `k8s/` directory for Kubernetes deployment manifests (if available).

## Performance Tips

1. **Use BuildKit** for faster builds:
   ```bash
   DOCKER_BUILDKIT=1 docker-compose build
   ```

2. **Layer Caching**: Order Dockerfile commands from least to most frequently changing

3. **Shared Memory**: Increase for DataLoader workers:
   ```yaml
   shm_size: '16gb'
   ```

4. **GPU Memory**: Use mixed precision training to reduce VRAM usage

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Docs](https://docs.docker.com/compose/)
- [Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## Support

For Docker-related issues:
1. Check logs: `docker-compose logs`
2. Open an [issue](https://github.com/Tommaso-R-Marena/QuantumFold-Advantage/issues)
3. Contact: marena@cua.edu
