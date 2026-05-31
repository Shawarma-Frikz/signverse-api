from fastapi import APIRouter, Depends, status, BackgroundTasks, Request
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.schemas.user import (
    UserRegister, UserLogin, TokenResponse,
    UserResponse, UserUpdate, RefreshRequest,
    ResendVerificationRequest
)
from app.services import auth as auth_service
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    data: UserRegister,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    await auth_service.register_user(db, data, background_tasks)
    return {
        "message": "Account created. Please check your email to verify your account."
    }


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, data: UserLogin, db: Session = Depends(get_db)):
    return auth_service.login_user(db, data)


@router.post("/refresh", response_model=TokenResponse)
def refresh(data: RefreshRequest, db: Session = Depends(get_db)):
    return auth_service.refresh_tokens(db, data.refresh_token)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserResponse)
def update_me(
    data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return auth_service.update_user(db, current_user, data)

@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    return auth_service.verify_email(db, token)

@router.post("/resend-verification")
async def resend_verification(
    data: ResendVerificationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    return await auth_service.resend_verification(db, data.email, background_tasks)