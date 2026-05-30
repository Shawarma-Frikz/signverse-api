from fastapi import APIRouter, Depends, status, BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.user import UserRegister, UserLogin, TokenResponse
from app.services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


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
def login(data: UserLogin, db: Session = Depends(get_db)):
    return auth_service.login_user(db, data)


@router.get("/me")
async def get_me():
    return {"message": "profile endpoint — coming next"}