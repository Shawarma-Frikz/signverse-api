from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserRegister(BaseModel):
    email:        EmailStr
    password:     str
    display_name: Optional[str] = None

class UserLogin(BaseModel):
    email:    EmailStr
    password: str

class UserUpdate(BaseModel):
    display_name:       Optional[str] = None
    bio:                Optional[str] = None
    avatar_url:         Optional[str] = None
    preferred_language: Optional[str] = None

class UserResponse(BaseModel):
    id:                 int
    email:              str
    display_name:       Optional[str]
    bio:                Optional[str]
    avatar_url:         Optional[str]
    preferred_language: str
    is_verified:        bool
    created_at:         datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str

class ResendVerificationRequest(BaseModel):
    email: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token:        str
    new_password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str