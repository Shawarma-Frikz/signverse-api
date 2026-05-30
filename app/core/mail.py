import resend
from itsdangerous import URLSafeTimedSerializer
from app.core.config import settings

resend.api_key = settings.resend_api_key

serializer = URLSafeTimedSerializer(settings.secret_key)


def generate_verification_token(email: str) -> str:
    return serializer.dumps(email, salt="email-verification")


def verify_token(token: str, max_age: int = 86400) -> str | None:
    try:
        return serializer.loads(token, salt="email-verification", max_age=max_age)
    except Exception:
        return None


def generate_reset_token(email: str) -> str:
    return serializer.dumps(email, salt="password-reset")


def verify_reset_token(token: str) -> str | None:
    try:
        return serializer.loads(token, salt="password-reset", max_age=3600)
    except Exception:
        return None


async def send_verification_email(email: str, token: str):
    verify_url = f"{settings.frontend_url}/api/v1/auth/verify-email?token={token}"

    resend.Emails.send({
        "from": settings.mail_from,
        "to": email,
        "subject": "Verify your SignVerse account",
        "html": f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #f4f4f4; padding: 40px;">
          <div style="max-width: 500px; margin: 0 auto; background: white;
                      border-radius: 12px; padding: 40px;">
            <h1 style="color: #1E3A5F;">SignVerse</h1>
            <p style="color: #525252; font-size: 16px;">
              Thanks for signing up. Please verify your email address.
            </p>
            <a href="{verify_url}"
               style="display: inline-block; margin: 24px 0; padding: 14px 28px;
                      background: #00BCD4; color: white; border-radius: 8px;
                      text-decoration: none; font-weight: 600;">
              Verify My Email
            </a>
            <p style="color: #A3A3A3; font-size: 13px;">
              This link expires in 24 hours.
            </p>
          </div>
        </body>
        </html>
        """
    })