from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.schemas.prediction import (
    AlphabetPredictRequest,
    AlphabetPredictResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from app.services import ml as ml_service
from app.models.user import User

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/alphabet", response_model=AlphabetPredictResponse)
def predict_alphabet(
    data: AlphabetPredictRequest,
    current_user: User = Depends(get_current_user)
):
    if len(data.landmarks) != 63:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected exactly 63 landmark values, got {len(data.landmarks)}"
        )
    try:
        return ml_service.predict_alphabet(data.landmarks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(
    data: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    feedback = ml_service.save_feedback(db, data, current_user)
    return {
        "message": "Feedback saved. Thank you for helping improve SignVerse.",
        "feedback_id": feedback.id
    }