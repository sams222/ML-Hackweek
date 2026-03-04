import uuid
import os
import asyncio
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import aiofiles
from app.models.schemas import UploadResponse
from app.jobs.job_store import job_store
from app.config import settings

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(settings.uploads_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, "input.mp4")

    # Stream to disk
    async with aiofiles.open(input_path, "wb") as f:
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large (max 500 MB)")
            await f.write(chunk)

    job_store.create(job_id)

    # Import here to avoid circular imports at module load
    from app.pipeline.coach_pipeline import run_pipeline
    background_tasks.add_task(run_pipeline, job_id, input_path)

    return UploadResponse(job_id=job_id)
