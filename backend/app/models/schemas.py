from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float


class PredictionResult(BaseModel):
    activity: str
    confidence: float
    next_action: str
    landmarks: List[Dict[str, float]] = []
    history: List[str] = []
    pose_detected: bool = True
    all_probabilities: Optional[Dict[str, float]] = None


class FrameAnalysis(BaseModel):
    activity: str
    confidence: float