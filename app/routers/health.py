from fastapi import APIRouter
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    except ImportError:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

    return HealthResponse(status="ok", gpu_available=gpu_available)
