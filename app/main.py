from fastapi import FastAPI

app = FastAPI(
    title="SignVerse API",
    description="Sign language translation backend",
    version="0.1.0",
)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}