from sqlalchemy.orm import Session
from fastapi import HTTPException, status, BackgroundTasks
from app.models.user import User
from app.schemas.user import UserRegister, UserLogin
from app.core.security import hash_password, verify_password, create_access_token, create_refresh_token
from app.core.mail import generate_verification_token, send_verification_email
from app.schemas.user import UserRegister, UserLogin, UserUpdate


async def register_user(
    db: Session,
    data: UserRegister,
    background_tasks: BackgroundTasks
) -> User:
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

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

    token = generate_verification_token(new_user.email)
    background_tasks.add_task(send_verification_email, new_user.email, token)

    return new_user


def login_user(db: Session, data: UserLogin) -> dict:
    # Find user by email
    user = db.query(User).filter(User.email == data.email).first()

    # Use same error for wrong email or wrong password
    # Never tell the user which one is wrong (security)
    invalid_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password"
    )

    if not user:
        raise invalid_error

    if not verify_password(data.password, user.hashed_password):
        raise invalid_error

    # Block unverified users with a clear message
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email before logging in"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )

    # Generate both tokens
    token_data = {"sub": str(user.id), "email": user.email}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

def update_user(db: Session, user: User, data: UserUpdate) -> User:
    # Only update fields that were actually sent
    if data.display_name is not None:
        user.display_name = data.display_name

    if data.preferred_language is not None:
        # Validate language is one we support
        allowed_languages = ["fr", "ar", "en"]
        if data.preferred_language not in allowed_languages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Language must be one of: {', '.join(allowed_languages)}"
            )
        user.preferred_language = data.preferred_language

    db.commit()
    db.refresh(user)
    return user