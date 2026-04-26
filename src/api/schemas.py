# src/api/schemas.py
from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    probability: float
    prediction: str
    confidence: float
    processing_time: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None