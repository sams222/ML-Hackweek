"""
OpenCV-based frame extraction and annotated video assembly.
"""
import os
import cv2
import numpy as np
from app.config import settings


def extract_frames(video_path: str) -> tuple[list[tuple[np.ndarray, int]], float]:
    """
    Extract every Nth frame from a video as RGB arrays.

    Returns (frames, source_fps) where frames is list of (rgb_frame, timestamp_ms).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_rate = settings.frame_sample_rate
    frames = []
    frame_idx = 0

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_idx / source_fps) * 1000)
            frames.append((rgb_frame, timestamp_ms))
        frame_idx += 1

    cap.release()
    return frames, source_fps


def _find_ffmpeg() -> str | None:
    """Find ffmpeg on disk, checking common winget/chocolatey install paths."""
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Common Windows install locations
    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"),
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    # Also glob for any version in the WinGet Packages dir
    import glob
    candidates += glob.glob(os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\*ffmpeg*\*\bin\ffmpeg.exe"
    ))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def assemble_annotated_video(pose_results: list[dict], output_path: str, fps: float = 2.0) -> str:
    """
    Write annotated frames to a browser-compatible H.264 MP4.
    Writes JPEG frames to a temp directory, then uses ffmpeg to encode.

    Returns the output path.
    """
    import subprocess
    import shutil
    import tempfile

    if not pose_results:
        raise ValueError("No frames to assemble")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cap at 90 frames max
    step = max(1, len(pose_results) // 90)
    sampled = pose_results[::step]

    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        # Write frames as numbered JPEGs to a temp dir
        tmp_dir = tempfile.mkdtemp(prefix="jora_frames_")
        try:
            for i, r in enumerate(sampled):
                bgr = cv2.cvtColor(r["annotated_frame"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

            result = subprocess.run(
                [ffmpeg, "-y",
                 "-framerate", str(fps),
                 "-i", os.path.join(tmp_dir, "%05d.jpg"),
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 output_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                return output_path
            else:
                print(f"[VideoService] ffmpeg failed: {result.stderr[:500]}")
        except Exception as exc:
            print(f"[VideoService] ffmpeg error: {exc}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Fallback: OpenCV mp4v (won't play in browser but at least won't hang)
    first_frame = sampled[0]["annotated_frame"]
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for r in sampled:
        bgr = cv2.cvtColor(r["annotated_frame"], cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()

    return output_path


def frames_to_pil(rgb_frames: list[np.ndarray], max_dim: int = 720):
    """Convert RGB numpy arrays to PIL Images for Gemini upload, downscaled for speed."""
    from PIL import Image
    pil_images = []
    for frame in rgb_frames:
        img = Image.fromarray(frame)
        # Downscale large frames to reduce Gemini upload time
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        pil_images.append(img)
    return pil_images
