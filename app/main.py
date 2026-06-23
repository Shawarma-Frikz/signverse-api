from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.routers import auth, predict, translations, users
from fastapi.staticfiles import StaticFiles
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup — loads model into memory
    print("Loading ML models...")
    from app.services import ml  # noqa — triggers module-level model loading
    print("ML models ready.")
    yield
    # Runs on shutdown
    print("Shutting down.")

# Create limiter — identifies users by IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # <-- ADDED: Connects the lifespan context manager
)

# Attach limiter to app state
app.state.limiter = limiter

# Handle rate limit exceeded with a clean JSON response
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")
app.include_router(translations.router,  prefix="/api/v1")
app.include_router(users.router,        prefix="/api/v1")

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": settings.version}

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")