from fastapi import APIRouter, HTTPException
from app.models.schemas import JobStatusResponse, JobResultResponse, JobStatus
from app.jobs.job_store import job_store

router = APIRouter()


def _get_job_or_404(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = _get_job_or_404(job_id)
    return JobStatusResponse(
        status=job.status,
        stage=job.stage,
        progress_pct=job.progress_pct,
        error=job.error,
    )


@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    job = _get_job_or_404(job_id)
    if job.status != JobStatus.completed:
        raise HTTPException(status_code=409, detail=f"Job is not completed (status: {job.status})")
    return JobResultResponse(feedback=job.feedback, output_urls=job.output_urls)
