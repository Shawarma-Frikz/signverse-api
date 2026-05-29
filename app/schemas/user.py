from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    display_name: str | None = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    display_name: str | None
    preferred_language: str
    created_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"