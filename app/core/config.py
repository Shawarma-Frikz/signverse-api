from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "SignVerse API"
    version: str = "0.1.0"
    debug: bool = False

    # Database
    database_url: str

    # JWT
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Email
    resend_api_key: str
    mail_from: str
    frontend_url: str

    class Config:
        env_file = ".env"

settings = Settings()