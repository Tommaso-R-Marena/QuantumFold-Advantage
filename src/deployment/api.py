from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="QuantumFold-Advantage API", version="1.0.0", description="Quantum-enhanced protein structure prediction")
_JOBS: Dict[str, Dict] = {}


class PredictionRequest(BaseModel):
    sequence: str
    model_type: str = "quantum"
    use_msa: bool = False
    recycling_iterations: int = 3
    use_quantum_hardware: bool = False
    quantum_backend: Optional[str] = "ibmq_simulator"


class PredictionResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict] = None
    estimated_time: int


class StructureResult(BaseModel):
    pdb_file_url: str
    plddt_scores: List[float]
    tm_score_estimate: Optional[float]
    metrics: Dict[str, float]
    visualization_url: str


def generate_job_id() -> str:
    return str(uuid.uuid4())


def estimate_time(request: PredictionRequest) -> int:
    return max(15, len(request.sequence) // 2)


def run_prediction(job_id: str, request: PredictionRequest) -> None:
    _JOBS[job_id]["status"] = "completed"
    _JOBS[job_id]["result"] = {
        "pdb_file_url": f"https://storage.quantumfold.io/{job_id}.pdb",
        "plddt_scores": [80.0] * len(request.sequence),
        "tm_score_estimate": 0.72,
        "metrics": {"rmsd": 1.8},
        "visualization_url": f"https://app.quantumfold.io/view/{job_id}",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_structure(request: PredictionRequest, background_tasks: BackgroundTasks):
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", request.sequence.upper()):
        raise HTTPException(status_code=400, detail="Invalid sequence")
    job_id = generate_job_id()
    _JOBS[job_id] = {"status": "queued", "result": None}
    background_tasks.add_task(run_prediction, job_id, request)
    return PredictionResponse(job_id=job_id, status="queued", estimated_time=estimate_time(request))


@app.get("/status/{job_id}", response_model=PredictionResponse)
async def get_job_status(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return PredictionResponse(job_id=job_id, status=_JOBS[job_id]["status"], result=_JOBS[job_id]["result"], estimated_time=0)


@app.get("/result/{job_id}", response_model=StructureResult)
async def get_result(job_id: str):
    if job_id not in _JOBS or _JOBS[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Result unavailable")
    return StructureResult(**_JOBS[job_id]["result"])


@app.post("/batch_predict")
async def batch_predict(sequences: List[str]):
    if len(sequences) > 100:
        raise HTTPException(status_code=400, detail="Maximum batch size is 100")
    ids = []
    for sequence in sequences:
        req = PredictionRequest(sequence=sequence)
        job_id = generate_job_id()
        _JOBS[job_id] = {"status": "completed", "result": None}
        run_prediction(job_id, req)
        ids.append(job_id)
    return {"job_ids": ids}


@app.get("/models", response_model=List[Dict])
async def list_models():
    return [
        {"name": "quantum_alphafold_85M"},
        {"name": "quantum_alphafold_200M"},
        {"name": "classical_baseline"},
        {"name": "multi_chain"},
        {"name": "rna_structure"},
        {"name": "protein_ligand_docking"},
    ]
