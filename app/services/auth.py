from sqlalchemy.orm import Session
from fastapi import HTTPException, status, BackgroundTasks
from app.models.user import User
from app.schemas.user import UserRegister
from app.core.security import hash_password
from app.core.mail import generate_verification_token, send_verification_email


async def register_user(
    db: Session,
    data: UserRegister,
    background_tasks: BackgroundTasks
) -> User:
    # Check if email already exists
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Hash password and create user
    new_user = User(
        email=data.email,
        hashed_password=hash_password(data.password),
        display_name=data.display_name,
        preferred_language="fr",
        is_verified=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Generate token and send verification email in the background
    # Background task means the API responds immediately without
    # waiting for the email to send
    token = generate_verification_token(new_user.email)
    background_tasks.add_task(send_verification_email, new_user.email, token)

    return new_user