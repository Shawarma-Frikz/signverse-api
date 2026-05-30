from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from itsdangerous import URLSafeTimedSerializer
from app.core.config import settings

# SMTP connection config
mail_config = ConnectionConfig(
    MAIL_USERNAME=settings.mail_username,
    MAIL_PASSWORD=settings.mail_password,
    MAIL_FROM=settings.mail_from,
    MAIL_PORT=settings.mail_port,
    MAIL_SERVER=settings.mail_server,
    MAIL_STARTTLS=settings.mail_starttls,
    MAIL_SSL_TLS=settings.mail_ssl_tls,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
)

fastmail = FastMail(mail_config)

# Token serializer — signs tokens with your SECRET_KEY
serializer = URLSafeTimedSerializer(settings.secret_key)


def generate_verification_token(email: str) -> str:
    """Generate a signed token that expires in 24 hours."""
    return serializer.dumps(email, salt="email-verification")


def verify_token(token: str, max_age: int = 86400) -> str | None:
    """
    Decode and validate a token.
    max_age=86400 = 24 hours.
    Returns the email if valid, None if expired or tampered.
    """
    try:
        email = serializer.loads(token, salt="email-verification", max_age=max_age)
        return email
    except Exception:
        return None


def generate_reset_token(email: str) -> str:
    """Generate a signed password reset token that expires in 1 hour."""
    return serializer.dumps(email, salt="password-reset")


def verify_reset_token(token: str) -> str | None:
    """
    Decode and validate a reset token.
    max_age=3600 = 1 hour.
    """
    try:
        email = serializer.loads(token, salt="password-reset", max_age=3600)
        return email
    except Exception:
        return None


async def send_verification_email(email: str, token: str):
    """Send the account verification email."""
    verify_url = f"{settings.frontend_url}/api/v1/auth/verify-email?token={token}"

    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background: #f4f4f4; padding: 40px;">
      <div style="max-width: 500px; margin: 0 auto; background: white;
                  border-radius: 12px; padding: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h1 style="color: #1E3A5F; margin-bottom: 8px;">SignVerse</h1>
        <p style="color: #525252; font-size: 16px;">
          Thanks for signing up. Please verify your email address to activate your account.
        </p>
        <a href="{verify_url}"
           style="display: inline-block; margin: 24px 0; padding: 14px 28px;
                  background: #00BCD4; color: white; border-radius: 8px;
                  text-decoration: none; font-weight: 600; font-size: 16px;">
          Verify My Email
        </a>
        <p style="color: #A3A3A3; font-size: 13px;">
          This link expires in 24 hours. If you didn't create an account, ignore this email.
        </p>
      </div>
    </body>
    </html>
    """

    message = MessageSchema(
        subject="Verify your SignVerse account",
        recipients=[email],
        body=html_content,
        subtype=MessageType.html,
    )

    await fastmail.send_message(message)