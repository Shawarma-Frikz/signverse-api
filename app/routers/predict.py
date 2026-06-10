from fastapi import APIRouter, Depends, HTTPException, status
from app.core.dependencies import get_current_user
from app.schemas.prediction import AlphabetPredictRequest, AlphabetPredictResponse
from app.services import ml as ml_service
from app.models.user import User

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "/alphabet",
    response_model=AlphabetPredictResponse
)
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
        result = ml_service.predict_alphabet(data.landmarks)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )