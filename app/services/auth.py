from sqlalchemy.orm import Session
from fastapi import HTTPException, status, BackgroundTasks
from app.models.user import User
from app.schemas.user import UserRegister, UserLogin, UserUpdate, RefreshRequest
from app.core.security import hash_password, verify_password, create_access_token, create_refresh_token
from app.core.mail import generate_verification_token, send_verification_email, verify_token, send_welcome_email
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token
)

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

def refresh_access_token(data: RefreshRequest) -> dict:
    # Decode the refresh token
    payload = decode_token(data.refresh_token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Make sure it's actually a refresh token, not an access token
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Issue a brand new access token with the same user data
    token_data = {
        "sub": payload.get("sub"),
        "email": payload.get("email")
    }
    new_access_token = create_access_token(token_data)

    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }

def verify_email(db: Session, token: str, background_tasks: BackgroundTasks) -> dict:
    email = verify_token(token)

    if email is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification link is invalid or has expired"
        )

    user = db.query(User).filter(User.email == email).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if user.is_verified:
        return {"message": "Email already verified. You can log in."}

    user.is_verified = True
    db.commit()

    # Send welcome email as background task
    background_tasks.add_task(send_welcome_email, user.email, user.display_name)

    return {"message": "Email verified successfully. You can now log in."}

async def resend_verification(
    db: Session,
    email: str,
    background_tasks: BackgroundTasks
) -> dict:
    user = db.query(User).filter(User.email == email).first()

    # Always return the same response whether user exists or not
    # Prevents user enumeration (attacker can't tell if email is registered)
    generic_response = {
        "message": "If that email is registered and unverified, a new verification link has been sent."
    }

    if user is None:
        return generic_response

    if user.is_verified:
        return {"message": "This email is already verified. You can log in."}

    # Generate fresh token and resend
    token = generate_verification_token(user.email)
    background_tasks.add_task(send_verification_email, user.email, token)

    return generic_response