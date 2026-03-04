from enum import Enum
from typing import Optional
from pydantic import BaseModel


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class PipelineStage(str, Enum):
    extracting_frames = "extracting_frames"
    pose_estimation = "pose_estimation"
    gemini_analysis = "gemini_analysis"
    tts_generation = "tts_generation"
    assembling_output = "assembling_output"


class KeyMoment(BaseModel):
    frame_index: int
    observation: str


class FeedbackResult(BaseModel):
    overall_summary: str
    form: list[str]
    movement: list[str]
    route_reading: list[str]
    key_moments: list[KeyMoment]
    encouragement: str


class OutputUrls(BaseModel):
    annotated_video: Optional[str] = None
    coaching_audio: Optional[str] = None


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    stage: Optional[PipelineStage] = None
    progress_pct: int = 0
    error: Optional[str] = None
    feedback: Optional[FeedbackResult] = None
    output_urls: Optional[OutputUrls] = None


class JobStatusResponse(BaseModel):
    status: JobStatus
    stage: Optional[PipelineStage] = None
    progress_pct: int = 0
    error: Optional[str] = None


class JobResultResponse(BaseModel):
    feedback: Optional[FeedbackResult] = None
    output_urls: Optional[OutputUrls] = None


class UploadResponse(BaseModel):
    job_id: str


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
