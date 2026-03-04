import threading
from typing import Optional
from app.models.schemas import JobRecord, JobStatus, PipelineStage, FeedbackResult, OutputUrls


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> JobRecord:
        record = JobRecord(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = record
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_stage(self, job_id: str, stage: PipelineStage, progress_pct: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.processing
                job.stage = stage
                job.progress_pct = progress_pct

    def complete(self, job_id: str, feedback: FeedbackResult, output_urls: OutputUrls) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.completed
                job.progress_pct = 100
                job.feedback = feedback
                job.output_urls = output_urls

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.failed
                job.error = error


job_store = JobStore()
