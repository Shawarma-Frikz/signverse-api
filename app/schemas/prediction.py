from pydantic import BaseModel
from typing import Optional

class AlphabetPredictRequest(BaseModel):
    landmarks: list[float]  # exactly 63 values

class PredictionResult(BaseModel):
    letter: str
    confidence: float

class AlphabetPredictResponse(BaseModel):
    letter:     str
    confidence: float
    top5:       list[PredictionResult]

class FeedbackRequest(BaseModel):
    model_type:      str            # "alphabet" or "word"
    predicted_label: str            # what the model predicted
    correct_label:   Optional[str] = None  # what it actually was
    confidence:      Optional[float] = None
    landmarks:       Optional[list[float]] = None  # the raw landmarks

class FeedbackResponse(BaseModel):
    id:              int
    model_type:      str
    predicted_label: str
    correct_label:   Optional[str]
    created_at:      str

    class Config:
        from_attributes = True