from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
async def register():
    return {"message": "register endpoint — coming in Sprint 2"}

@router.post("/login")
async def login():
    return {"message": "login endpoint — coming in Sprint 2"}

@router.get("/me")
async def get_me():
    return {"message": "profile endpoint — coming in Sprint 2"}