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
    quality_classification = "quality_classification"
    gemini_analysis = "gemini_analysis"
    tts_generation = "tts_generation"
    assembling_output = "assembling_output"


class KeyMoment(BaseModel):
    timestamp: str = "0:00"        # "M:SS" format, e.g. "0:12"
    timestamp_sec: float = 0.0     # seconds, used by frontend to seek video
    observation: str


class QualityFlags(BaseModel):
    arm_efficiency:      str  # "good" | "needs_work"
    hip_positioning:     str
    movement_smoothness: str
    hesitation:          str
    body_tension:        str
    range_of_motion:     str


class QualityResult(BaseModel):
    """Output of the climb quality classifier. Injected into Gemini prompt."""
    verdict:       str            # "good" | "bad"
    score:         float          # 0–100 technique score
    confidence:    float          # 0–100 model confidence
    feature_flags: QualityFlags


class FeedbackResult(BaseModel):
    overall_summary: str
    form: list[str]
    movement: list[str]
    route_reading: list[str]
    key_moments: list[KeyMoment]
    encouragement: str


class OutputUrls(BaseModel):
    annotated_video: Optional[str] = None
    coaching_audio:  Optional[str] = None


class JobRecord(BaseModel):
    job_id:      str
    status:      JobStatus = JobStatus.queued
    stage:       Optional[PipelineStage] = None
    progress_pct: int = 0
    error:       Optional[str] = None
    feedback:    Optional[FeedbackResult] = None
    output_urls: Optional[OutputUrls] = None


class JobStatusResponse(BaseModel):
    status:       JobStatus
    stage:        Optional[PipelineStage] = None
    progress_pct: int = 0
    error:        Optional[str] = None


class JobResultResponse(BaseModel):
    feedback:    Optional[FeedbackResult] = None
    output_urls: Optional[OutputUrls] = None


class UploadResponse(BaseModel):
    job_id: str


class HealthResponse(BaseModel):
    status:        str
    gpu_available: bool
