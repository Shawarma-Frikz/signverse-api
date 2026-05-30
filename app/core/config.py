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
    mail_username: str
    mail_password: str
    mail_from: str
    mail_server: str = "smtp.gmail.com"
    mail_port: int = 587
    mail_starttls: bool = True
    mail_ssl_tls: bool = False
    frontend_url: str

    class Config:
        env_file = ".env"

settings = Settings()