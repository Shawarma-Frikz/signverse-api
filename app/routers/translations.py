from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.schemas.translation import (
    TranslationCreate,
    TranslationResponse,
    TranslationListResponse,
)
from app.services import translation as translation_service

router = APIRouter(prefix="/translations", tags=["Translations"])


@router.post(
    "/",
    response_model=TranslationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_translation(
    data: TranslationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a completed translation session to history."""
    if data.input_type not in ("alphabet", "word"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="input_type must be 'alphabet' or 'word'",
        )
    if not data.result_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="result_text cannot be empty",
        )

    return translation_service.create_translation(db, data, current_user)


@router.get("/", response_model=TranslationListResponse)
def get_translations(
    skip:  int = Query(default=0,  ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch the authenticated user's translation history."""
    translations, total = translation_service.get_user_translations(
        db, current_user, skip=skip, limit=limit
    )
    return TranslationListResponse(translations=translations, total=total)


@router.delete("/{translation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_translation(
    translation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a specific translation from history."""
    deleted = translation_service.delete_translation(
        db, translation_id, current_user
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Translation not found",
        )