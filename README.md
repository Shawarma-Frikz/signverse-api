# SignVerse API

FastAPI backend for SignVerse, a real-time sign language translation app.

## Stack
- **FastAPI** — REST API framework
- **PostgreSQL** — hosted on Railway
- **SQLAlchemy** — ORM
- **Alembic** — database migrations
- **JWT** — authentication

## Local Setup

1. Clone the repo
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in your values
5. Run: `uvicorn app.main:app --reload`

API docs available at `http://localhost:8000/docs`

## Deployment
Deployed on Railway. Live at: `https://signverse-api.up.railway.app`