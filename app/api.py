from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import upload, jobs, health
import os


def create_app() -> FastAPI:
    app = FastAPI(title="Jora", version="0.1.0")

    app.include_router(upload.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(health.router)

    # Serve output files (annotated videos, coach videos, audio)
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

    # Serve frontend static files
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def root():
        return FileResponse("static/index.html")

    return app
