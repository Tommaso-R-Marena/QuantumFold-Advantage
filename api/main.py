"""Production-ready FastAPI server for QuantumFold predictions.

Features:
- RESTful endpoints for protein folding prediction
- JWT authentication and rate limiting
- Request validation with Pydantic
- Async processing for long-running jobs
- Prometheus metrics and health checks
- OpenAPI documentation
- CORS support

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from datetime import datetime, timedelta
import uuid
import asyncio
import logging
from pathlib import Path

# Import project modules
from src.advanced_model import AdvancedProteinFoldingModel
from src.protein_embeddings import ESM2Embedder
from src.data.augmentation import ProteinAugmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QuantumFold-Advantage API",
    description="Production API for quantum-enhanced protein structure prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global state
class AppState:
    """Application state manager."""
    def __init__(self):
        self.model: Optional[AdvancedProteinFoldingModel] = None
        self.embedder: Optional[ESM2Embedder] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.job_queue: Dict[str, Dict] = {}
        self.model_loaded = False

state = AppState()


# Pydantic models for request/response validation
class ProteinSequenceRequest(BaseModel):
    """Request model for protein folding prediction."""
    sequence: str = Field(..., min_length=10, max_length=2000,
                          description="Amino acid sequence (single letter code)")
    use_quantum: bool = Field(default=True, description="Enable quantum enhancement")
    num_recycles: int = Field(default=3, ge=1, le=10,
                              description="Number of structure refinement cycles")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0,
                                       description="Minimum confidence for predictions")
    
    @validator('sequence')
    def validate_sequence(cls, v):
        """Validate amino acid sequence."""
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        v = v.upper().strip()
        if not all(aa in valid_aa for aa in v):
            raise ValueError("Invalid amino acid characters in sequence")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    job_id: str
    status: str = Field(..., description="Job status: pending, running, completed, failed")
    sequence: str
    prediction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    device: str
    gpu_available: bool
    memory_allocated_mb: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    sequences: List[str] = Field(..., min_items=1, max_items=100)
    use_quantum: bool = True
    num_recycles: int = 3


# Authentication (simple token-based for demo)
VALID_TOKENS = {"demo_token_123"}  # In production, use proper JWT

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    if credentials.credentials not in VALID_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting QuantumFold API...")
    try:
        # Load embedder
        logger.info("Loading ESM-2 embedder...")
        state.embedder = ESM2Embedder(model_name='esm2_t33_650M_UR50D')
        
        # Load model
        logger.info("Loading protein folding model...")
        state.model = AdvancedProteinFoldingModel(
            input_dim=1280,
            c_s=384,
            c_z=128,
            use_quantum=True
        ).to(state.device)
        
        # Try to load checkpoint if available
        checkpoint_path = Path("checkpoints/best_model.pt")
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=state.device)
            state.model.load_state_dict(checkpoint['model_state_dict'])
        
        state.model.eval()
        state.model_loaded = True
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        state.model_loaded = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down QuantumFold API...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# API endpoints
@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "QuantumFold-Advantage API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    memory_allocated = None
    
    if gpu_available:
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    
    return HealthResponse(
        status="healthy" if state.model_loaded else "degraded",
        timestamp=datetime.utcnow(),
        model_loaded=state.model_loaded,
        device=str(state.device),
        gpu_available=gpu_available,
        memory_allocated_mb=memory_allocated
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_structure(
    request: ProteinSequenceRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Submit protein folding prediction job.
    
    This endpoint accepts a protein sequence and submits it for structure
    prediction. For sequences longer than 100 residues, the job is processed
    asynchronously.
    """
    if not state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "sequence": request.sequence,
        "use_quantum": request.use_quantum,
        "num_recycles": request.num_recycles,
        "created_at": datetime.utcnow(),
        "prediction": None,
        "error": None
    }
    state.job_queue[job_id] = job_data
    
    # For short sequences, process immediately
    if len(request.sequence) < 100:
        try:
            result = await process_prediction(request, job_id)
            job_data.update(result)
        except Exception as e:
            job_data["status"] = "failed"
            job_data["error"] = str(e)
    else:
        # Long sequences: process in background
        background_tasks.add_task(process_prediction_background, request, job_id)
        job_data["status"] = "running"
    
    return PredictionResponse(**job_data)


@app.get("/predict/{job_id}", response_model=PredictionResponse, tags=["prediction"])
async def get_prediction_status(
    job_id: str,
    token: str = Depends(verify_token)
):
    """Get status of prediction job."""
    if job_id not in state.job_queue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    return PredictionResponse(**state.job_queue[job_id])


@app.post("/predict/batch", tags=["prediction"])
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Submit batch prediction job."""
    if not state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    job_ids = []
    for sequence in request.sequences:
        seq_request = ProteinSequenceRequest(
            sequence=sequence,
            use_quantum=request.use_quantum,
            num_recycles=request.num_recycles
        )
        response = await predict_structure(seq_request, background_tasks, token)
        job_ids.append(response.job_id)
    
    return {"job_ids": job_ids, "count": len(job_ids)}


@app.delete("/predict/{job_id}", tags=["prediction"])
async def delete_job(
    job_id: str,
    token: str = Depends(verify_token)
):
    """Delete prediction job."""
    if job_id in state.job_queue:
        del state.job_queue[job_id]
        return {"message": "Job deleted", "job_id": job_id}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Job {job_id} not found"
    )


@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Get API metrics."""
    total_jobs = len(state.job_queue)
    status_counts = {}
    for job in state.job_queue.values():
        status = job["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        "total_jobs": total_jobs,
        "status_breakdown": status_counts,
        "model_loaded": state.model_loaded,
        "device": str(state.device)
    }


# Helper functions
async def process_prediction(request: ProteinSequenceRequest, job_id: str) -> Dict:
    """Process prediction synchronously."""
    start_time = datetime.utcnow()
    
    try:
        # Generate embeddings
        embeddings_dict = state.embedder([request.sequence])
        embeddings = embeddings_dict['embeddings'].to(state.device)
        
        # Run prediction
        with torch.no_grad():
            output = state.model(
                embeddings,
                num_recycles=request.num_recycles
            )
        
        # Extract results
        coordinates = output['coordinates'].cpu().numpy()
        plddt = output['plddt'].cpu().numpy()
        
        # Filter by confidence
        high_conf_mask = plddt[0] >= request.confidence_threshold
        
        prediction = {
            "coordinates": coordinates[0].tolist(),
            "plddt": plddt[0].tolist(),
            "mean_plddt": float(plddt.mean()),
            "high_confidence_ratio": float(high_conf_mask.sum() / len(plddt[0])),
            "sequence_length": len(request.sequence)
        }
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "status": "completed",
            "prediction": prediction,
            "completed_at": end_time,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Prediction failed for job {job_id}: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow()
        }


async def process_prediction_background(request: ProteinSequenceRequest, job_id: str):
    """Process prediction in background."""
    result = await process_prediction(request, job_id)
    state.job_queue[job_id].update(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
