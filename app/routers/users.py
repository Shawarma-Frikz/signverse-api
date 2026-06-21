from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.schemas.user import UserResponse
import base64
import io
from PIL import Image

router = APIRouter(prefix="/users", tags=["Users"])

MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
ALLOWED_TYPES  = {"image/jpeg", "image/png", "image/webp"}
OUTPUT_SIZE    = (400, 400)        # Resize to 400×400


@router.post("/avatar", response_model=UserResponse)
async def upload_avatar(
    file:         UploadFile = File(...),
    current_user: User       = Depends(get_current_user),
    db:           Session    = Depends(get_db),
):
    """Upload and store avatar as base64 data URI."""
    # Validate type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only JPEG, PNG, and WebP images are allowed.",
        )

    # Read and validate size
    data = await file.read()
    if len(data) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image must be under 5MB.",
        )

    # Resize to square 400×400 and compress
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = _crop_square(img)
    img = img.resize(OUTPUT_SIZE, Image.LANCZOS)

    # Save as JPEG to buffer
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    # Store as base64 data URI (no external storage needed)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    avatar_url = f"data:image/jpeg;base64,{b64}"

    # Save to DB
    current_user.avatar_url = avatar_url
    db.commit()
    db.refresh(current_user)

    return current_user


def _crop_square(img: Image.Image) -> Image.Image:
    """Center-crop image to a square."""
    w, h  = img.size
    side  = min(w, h)
    left  = (w - side) // 2
    top   = (h - side) // 2
    return img.crop((left, top, left + side, top + side))