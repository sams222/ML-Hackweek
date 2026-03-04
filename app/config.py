from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    mock_tts: bool = True

    uploads_dir: str = "uploads"
    outputs_dir: str = "outputs"
    mediapipe_model_path: str = "models/pose_landmarker_full.task"

    frame_sample_rate: int = 15  # every Nth frame (30fps video → 2fps extraction)
    key_frame_count: int = 6
    pose_visibility_threshold: float = 0.5
    pose_confidence_threshold: float = 0.5


settings = Settings()
