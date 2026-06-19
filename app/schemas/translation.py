from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TranslationCreate(BaseModel):
    input_type:     str           # "alphabet" or "word"
    detected_signs: str           # comma-separated signs e.g. "H,E,L,L,O"
    result_text:    str           # final built sentence e.g. "HELLO"
    confidence:     Optional[float] = None
    duration_ms:    Optional[int]   = None

class TranslationResponse(BaseModel):
    id:             int
    input_type:     str
    detected_signs: str
    result_text:    str
    confidence:     Optional[float]
    duration_ms:    Optional[int]
    created_at:     datetime

    class Config:
        from_attributes = True

class TranslationListResponse(BaseModel):
    translations: list[TranslationResponse]
    total:        int