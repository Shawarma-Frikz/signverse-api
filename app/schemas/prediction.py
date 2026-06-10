from pydantic import BaseModel

class AlphabetPredictRequest(BaseModel):
    landmarks: list[float]  # exactly 63 values

class PredictionResult(BaseModel):
    letter: str
    confidence: float

class AlphabetPredictResponse(BaseModel):
    letter:     str
    confidence: float
    top5:       list[PredictionResult]