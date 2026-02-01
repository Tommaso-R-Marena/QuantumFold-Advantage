# QuantumFold-Advantage REST API

Production-ready REST API for protein structure prediction using quantum-enhanced deep learning.

## Features

- **RESTful API**: Standard HTTP endpoints for predictions
- **Authentication**: Token-based authentication (JWT ready)
- **Async Processing**: Background jobs for long sequences
- **Validation**: Request validation with Pydantic
- **Monitoring**: Health checks and metrics endpoints
- **Documentation**: Auto-generated OpenAPI docs
- **CORS**: Cross-origin resource sharing support

## Quick Start

### Installation

```bash
# Install API dependencies
pip install -r api/requirements.txt

# Install QuantumFold
pip install -e .
```

### Running Locally

```bash
# Development server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production server (4 workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Build image
docker build -t quantumfold-api -f api/Dockerfile .

# Run container
docker run -p 8000:8000 --gpus all quantumfold-api
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict Structure

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer demo_token_123" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",
    "use_quantum": true,
    "num_recycles": 3,
    "confidence_threshold": 0.7
  }'
```

### Get Job Status

```bash
curl http://localhost:8000/predict/{job_id} \
  -H "Authorization: Bearer demo_token_123"
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Authorization: Bearer demo_token_123" \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MKTAYIAKQRQISFVK",
      "SHFSRQLEERLGLIEV"
    ],
    "use_quantum": true
  }'
```

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Authentication

All prediction endpoints require authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer YOUR_TOKEN
```

For production, replace the simple token validation with proper JWT authentication.

## Response Format

### Successful Prediction

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",
  "prediction": {
    "coordinates": [[x1, y1, z1], [x2, y2, z2], ...],
    "plddt": [90.5, 88.3, ...],
    "mean_plddt": 87.2,
    "high_confidence_ratio": 0.92,
    "sequence_length": 54
  },
  "created_at": "2026-02-01T06:00:00Z",
  "completed_at": "2026-02-01T06:00:15Z",
  "processing_time": 15.234
}
```

## Configuration

Environment variables:

```bash
# Server
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model
MODEL_CHECKPOINT=checkpoints/best_model.pt
ESM_MODEL=esm2_t33_650M_UR50D

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    image: quantumfold-api:latest
    ports:
      - "8000:8000"
    environment:
      - API_WORKERS=4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Kubernetes

See `api/k8s/` for Kubernetes deployment manifests.

## Monitoring

Metrics endpoint:

```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "total_jobs": 42,
  "status_breakdown": {
    "completed": 35,
    "running": 3,
    "pending": 2,
    "failed": 2
  },
  "model_loaded": true,
  "device": "cuda:0"
}
```

## Rate Limiting

For production deployment, implement rate limiting using:

- **slowapi**: Python rate limiting
- **nginx**: Upstream rate limiting
- **API Gateway**: Cloud-based rate limiting

## Security Considerations

1. **Replace demo tokens** with proper JWT authentication
2. **Enable HTTPS** in production
3. **Configure CORS** appropriately
4. **Implement rate limiting**
5. **Add input sanitization**
6. **Enable request logging**
7. **Use secrets management** for sensitive data

## Performance Tips

1. **GPU acceleration**: Deploy on GPU-enabled instances
2. **Model caching**: Keep model loaded in memory
3. **Batch processing**: Use batch endpoints for multiple sequences
4. **Async processing**: Long sequences processed in background
5. **Load balancing**: Deploy multiple workers

## License

MIT License - see LICENSE file
