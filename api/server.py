"""FastAPI server for QuantumFold predictions.

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import time
import uuid
from typing import Optional, Dict, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import torch
import numpy as np

from src.pipeline import QuantumFoldPipeline
from src.benchmarks import ProteinStructureEvaluator


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for structure prediction."""
    sequence: str = Field(..., description="Amino acid sequence", min_length=10, max_length=1000)
    use_quantum: bool = Field(True, description="Use quantum layers")
    save_pdb: bool = Field(True, description="Save structure as PDB file")
    
    @validator('sequence')
    def validate_sequence(cls, v):
        """Validate amino acid sequence."""
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        v = v.upper().strip()
        if not all(aa in valid_aa for aa in v):
            raise ValueError('Sequence contains invalid amino acids')
        return v


class PredictionResponse(BaseModel):
    """Response model for structure prediction."""
    job_id: str
    status: str
    sequence_length: int
    confidence: Optional[float] = None
    pdb_url: Optional[str] = None
    coordinates: Optional[List[List[float]]] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    device: str
    quantum_enabled: bool
    model_loaded: bool


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float
    message: str


# Initialize FastAPI app
app = FastAPI(
    title="QuantumFold-Advantage API",
    description="Hybrid quantum-classical protein structure prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.pipeline: Optional[QuantumFoldPipeline] = None
        self.evaluator = ProteinStructureEvaluator()
        self.jobs: Dict[str, Dict] = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path('outputs/api')
        self.output_dir.mkdir(parents=True, exist_ok=True)

state = AppState()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        checkpoint_path = 'checkpoints/quantumfold_best.pt'
        if Path(checkpoint_path).exists():
            state.pipeline = QuantumFoldPipeline(
                checkpoint=checkpoint_path,
                use_quantum=True,
                device=state.device
            )
            print(f"Model loaded successfully on {state.device}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("API will run without pre-trained model")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "QuantumFold-Advantage API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=state.device,
        quantum_enabled=True,
        model_loaded=state.pipeline is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_structure(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Predict protein structure from sequence.
    
    Args:
        request: Prediction request with sequence and options
        background_tasks: FastAPI background tasks
        
    Returns:
        Prediction response with job ID
    """
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Create job
    job_id = str(uuid.uuid4())
    job_data = {
        'job_id': job_id,
        'status': 'queued',
        'sequence': request.sequence,
        'sequence_length': len(request.sequence),
        'use_quantum': request.use_quantum,
        'save_pdb': request.save_pdb,
        'created_at': datetime.now().isoformat(),
        'progress': 0.0
    }
    state.jobs[job_id] = job_data
    
    # Queue prediction
    background_tasks.add_task(run_prediction, job_id)
    
    return PredictionResponse(
        job_id=job_id,
        status='queued',
        sequence_length=len(request.sequence),
        created_at=job_data['created_at']
    )


async def run_prediction(job_id: str):
    """Run prediction in background.
    
    Args:
        job_id: Job identifier
    """
    job = state.jobs[job_id]
    
    try:
        # Update status
        job['status'] = 'running'
        job['progress'] = 0.1
        
        # Run prediction
        sequence = job['sequence']
        job['progress'] = 0.3
        
        results = state.pipeline.predict(sequence)
        job['progress'] = 0.8
        
        # Extract results
        coordinates = results['coordinates']
        confidence = float(results.get('confidence', 0.0))
        
        job['coordinates'] = coordinates.tolist()
        job['confidence'] = confidence
        
        # Save PDB if requested
        if job['save_pdb']:
            pdb_path = state.output_dir / f"{job_id}.pdb"
            state.pipeline.save_structure(
                coordinates,
                str(pdb_path),
                sequence=sequence
            )
            job['pdb_path'] = str(pdb_path)
            job['pdb_url'] = f"/download/{job_id}.pdb"
        
        job['progress'] = 1.0
        job['status'] = 'completed'
        job['completed_at'] = datetime.now().isoformat()
        
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        job['progress'] = 0.0


@app.get("/status/{job_id}", response_model=PredictionResponse, tags=["Prediction"])
async def get_prediction_status(job_id: str):
    """Get prediction job status.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Prediction status
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = state.jobs[job_id]
    
    return PredictionResponse(
        job_id=job_id,
        status=job['status'],
        sequence_length=job['sequence_length'],
        confidence=job.get('confidence'),
        pdb_url=job.get('pdb_url'),
        coordinates=job.get('coordinates'),
        created_at=job['created_at'],
        completed_at=job.get('completed_at'),
        error=job.get('error')
    )


@app.get("/download/{filename}", tags=["Files"])
async def download_file(filename: str):
    """Download PDB file.
    
    Args:
        filename: File name
        
    Returns:
        File response
    """
    file_path = state.output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type='chemical/x-pdb',
        filename=filename
    )


@app.get("/jobs", tags=["Jobs"])
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """List all jobs.
    
    Args:
        limit: Maximum number of jobs to return
        status: Filter by status (queued, running, completed, failed)
        
    Returns:
        List of jobs
    """
    jobs = list(state.jobs.values())
    
    if status:
        jobs = [j for j in jobs if j['status'] == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    return {
        'total': len(jobs),
        'jobs': jobs[:limit]
    }


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Delete a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Success message
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = state.jobs[job_id]
    
    # Delete PDB file if exists
    if 'pdb_path' in job:
        pdb_path = Path(job['pdb_path'])
        if pdb_path.exists():
            pdb_path.unlink()
    
    # Delete job
    del state.jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_prediction(
    predicted_file: UploadFile = File(...),
    ground_truth_file: UploadFile = File(...)
):
    """Evaluate prediction against ground truth.
    
    Args:
        predicted_file: Predicted structure PDB file
        ground_truth_file: Ground truth structure PDB file
        
    Returns:
        Evaluation metrics
    """
    try:
        # Save uploaded files
        pred_path = state.output_dir / f"temp_pred_{uuid.uuid4()}.pdb"
        true_path = state.output_dir / f"temp_true_{uuid.uuid4()}.pdb"
        
        with open(pred_path, 'wb') as f:
            f.write(await predicted_file.read())
        
        with open(true_path, 'wb') as f:
            f.write(await ground_truth_file.read())
        
        # Load structures (simplified - would use proper PDB parser)
        # For now, return mock metrics
        from src.data import load_protein_structure
        
        coords_pred = load_protein_structure(str(pred_path))
        coords_true = load_protein_structure(str(true_path))
        
        # Evaluate
        metrics = state.evaluator.evaluate_structure(
            coords_pred,
            coords_true,
            sequence_length=len(coords_true)
        )
        
        # Cleanup
        pred_path.unlink()
        true_path.unlink()
        
        return metrics.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
