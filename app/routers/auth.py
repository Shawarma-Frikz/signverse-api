from fastapi import APIRouter, Depends, status, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.core.mail import verify_reset_token
from app.schemas.user import (
    UserRegister, UserLogin, TokenResponse,
    UserResponse, UserUpdate, RefreshRequest,
    ResendVerificationRequest, ForgotPasswordRequest,
    ResetPasswordRequest
)
from app.services import auth as auth_service
from app.models.user import User
import os

router = APIRouter(prefix="/auth", tags=["Authentication"])
limiter = Limiter(key_func=get_remote_address)

# Jinja2 templates
TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "templates"
)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str):
    email = verify_reset_token(token)
    token_valid = email is not None

    return templates.TemplateResponse(
        request=request,
        name="reset_password_page.html",
        context={
            "token": token,
            "token_valid": token_valid,
        }
    )


# ── All your existing endpoints below unchanged ───────────────────
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
def verify_email(
    token: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    return auth_service.verify_email(db, token, background_tasks)


@router.post("/resend-verification")
async def resend_verification(
    data: ResendVerificationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    return await auth_service.resend_verification(
        db, data.email, background_tasks
    )


@router.post("/forgot-password")
async def forgot_password(
    data: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    return await auth_service.forgot_password(
        db, data.email, background_tasks
    )


@router.post("/reset-password")
def reset_password(
    data: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    return auth_service.reset_password(db, data)