from sqlalchemy.orm import Session
from app.models.translation import Translation
from app.models.user import User
from app.schemas.translation import TranslationCreate

def create_translation(
    db: Session,
    data: TranslationCreate,
    user: User,
) -> Translation:
    translation = Translation(
        user_id        = user.id,
        input_type     = data.input_type,
        detected_signs = data.detected_signs,
        result_text    = data.result_text,
        confidence     = data.confidence,
        duration_ms    = data.duration_ms,
    )
    db.add(translation)
    db.commit()
    db.refresh(translation)
    return translation

def get_user_translations(
    db: Session,
    user: User,
    skip: int = 0,
    limit: int = 20,
) -> tuple[list[Translation], int]:
    total = db.query(Translation)\
        .filter(Translation.user_id == user.id)\
        .count()

    translations = db.query(Translation)\
        .filter(Translation.user_id == user.id)\
        .order_by(Translation.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

    return translations, total

def delete_translation(
    db: Session,
    translation_id: int,
    user: User,
) -> bool:
    translation = db.query(Translation)\
        .filter(
            Translation.id == translation_id,
            Translation.user_id == user.id,
        ).first()

    if not translation:
        return False

    db.delete(translation)
    db.commit()
    return True