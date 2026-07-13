"""
ORBITGUARD — FastAPI Backend
REST API serving all 3 AI modules.
"""

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pipeline import get_pipeline

app = FastAPI(
    title="ORBITGUARD API",
    description="AI-powered LEO debris avoidance system",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    get_pipeline()   # load models once


# ── REQUEST SCHEMAS ───────────────────────────────────────

class DetectRequest(BaseModel):
    sequences: List[List[List[float]]]

class PredictRequest(BaseModel):
    state0: List[float]
    dt_min: float = 30.0

class AvoidRequest(BaseModel):
    sc_state:      List[float]
    debris_states: List[List[float]]

class PipelineRequest(BaseModel):
    sc_state:      List[float]
    debris_states: List[List[float]]
    sequences:     List[List[List[float]]]


# ── ENDPOINTS ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "system":  "ORBITGUARD",
        "version": "1.0.0",
        "status":  "online",
        "modules": ["CNN+LSTM", "PINN", "PPO"],
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    p = get_pipeline()
    return {
        "status":  "healthy",
        "objects": len(p.objects),
        "metrics": {
            "det_f1":       0.9677,
            "pinn_rmse_km": 48.27,
            "drl_success":  1.0,
            "latency_ms":   131.7,
        },
    }


@app.get("/objects")
def get_objects():
    """
    Returns all tracked LEO objects for 3D visualisation.
    Called once by React frontend on load.
    """
    p = get_pipeline()
    return {
        "objects":      p.objects,
        "conjunctions": p.conjunctions,
        "meta": {
            "total": len(p.objects),
            "high":  sum(1 for o in p.objects
                         if o["risk"] == "HIGH"),
            "med":   sum(1 for o in p.objects
                         if o["risk"] == "MED"),
            "low":   sum(1 for o in p.objects
                         if o["risk"] == "LOW"),
            "conjunction_events": 209285,
            "trajectory_rows":    18125348,
            "pinn_rmse_km":       48.27,
            "det_f1":             0.9677,
            "drl_success":        1.0,
            "latency_ms":         131.7,
        },
    }


@app.post("/detect")
def detect(req: DetectRequest):
    """MODULE 1 — CNN+LSTM risk classification."""
    t0 = time.time()
    p  = get_pipeline()
    try:
        results = p.detect(req.sequences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "detections":   results,
        "latency_ms":   round((time.time()-t0)*1000, 2),
        "model":        "CNN+LSTM+Attention",
        "high_risk_f1": 0.9677,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """MODULE 2 — PINN trajectory prediction."""
    t0 = time.time()
    p  = get_pipeline()
    try:
        result = p.predict(req.state0, req.dt_min)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        **result,
        "latency_ms": round((time.time()-t0)*1000, 2),
        "model":      "PINN (Physics-Informed NN)",
    }


@app.post("/avoid")
def avoid(req: AvoidRequest):
    """MODULE 3 — PPO avoidance maneuver."""
    t0 = time.time()
    p  = get_pipeline()
    try:
        result = p.avoid(req.sc_state, req.debris_states)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        **result,
        "latency_ms":   round((time.time()-t0)*1000, 2),
        "model":        "PPO ActorCritic",
        "success_rate": 1.0,
    }


@app.post("/pipeline")
def run_pipeline(req: PipelineRequest):
    """Full ORBITGUARD pipeline — Detection→Prediction→Avoidance."""
    t0 = time.time()
    p  = get_pipeline()

    detections = p.detect(req.sequences)
    high_states = [
        req.debris_states[i]
        for i, d in enumerate(detections)
        if d["risk"] == "HIGH"
        and i < len(req.debris_states)
    ]
    predictions = [
        p.predict(s, 30.0) for s in high_states[:5]
    ]
    avoid_states = (
        high_states if high_states
        else req.debris_states[:5]
    )
    maneuver = p.avoid(req.sc_state, avoid_states)

    return {
        "detections":  detections,
        "predictions": predictions,
        "maneuver":    maneuver,
        "high_count":  len(high_states),
        "latency_ms":  round((time.time()-t0)*1000, 2),
        "pipeline":    "CNN+LSTM → PINN → PPO",
    }