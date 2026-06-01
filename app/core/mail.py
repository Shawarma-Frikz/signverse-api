import resend
from itsdangerous import URLSafeTimedSerializer
from app.core.config import settings
import os

resend.api_key = settings.resend_api_key

serializer = URLSafeTimedSerializer(settings.secret_key)

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


def _load_template(filename: str) -> str:
    with open(os.path.join(TEMPLATES_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


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
    html = _load_template("verification_email.html").replace("{{verify_url}}", verify_url)

    resend.Emails.send({
        "from": settings.mail_from,
        "to": email,
        "subject": "Verify your SignVerse account",
        "html": html,
    })

async def send_welcome_email(email: str, display_name: str):
    name = display_name or email.split("@")[0]
    html = _load_template("welcome_email.html").replace("{{display_name}}", name)

    resend.Emails.send({
        "from": settings.mail_from,
        "to": email,
        "subject": f"Welcome to SignVerse, {name} 👋",
        "html": html,
    })