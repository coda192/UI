from fastapi import FastAPI
from backend.api.v1.router import api_router

app = FastAPI(
    title="AKSIS Prototype API",
    description="Stable API contract for the AKSIS ML Framework UI Prototype",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}
