"""
Pipeline orchestrator: runs all services in sequence, updates job store at each stage.
Designed to run in a FastAPI BackgroundTask (asyncio event loop thread pool).
"""
import asyncio
import os
import time
import traceback
from app.config import settings
from app.jobs.job_store import job_store
from app.models.schemas import PipelineStage, OutputUrls


async def run_pipeline(job_id: str, input_video_path: str) -> None:
    output_dir = os.path.join(settings.outputs_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.monotonic()

    def _log(msg):
        elapsed = time.monotonic() - t_start
        print(f"[Pipeline {elapsed:6.1f}s] {msg}")

    try:
        # ── Stage 1: Extract frames ──────────────────────────────────────
        job_store.update_stage(job_id, PipelineStage.extracting_frames, 5)
        _log("Extracting frames...")
        from app.services.video_service import extract_frames
        frames, source_fps = await asyncio.to_thread(extract_frames, input_video_path)
        _log(f"Extracted {len(frames)} frames (source {source_fps:.0f}fps, sample rate {settings.frame_sample_rate})")

        if not frames:
            raise RuntimeError("No frames could be extracted from the video.")

        output_fps = source_fps / settings.frame_sample_rate

        # ── Stage 2: Pose estimation ─────────────────────────────────────
        job_store.update_stage(job_id, PipelineStage.pose_estimation, 20)
        _log("Running pose estimation...")
        from app.services.pose_service import process_frames, select_key_frames, compute_angle_stats
        pose_results = await asyncio.to_thread(process_frames, frames)
        detected_count = sum(1 for r in pose_results if r["detected"])
        _log(f"Pose done: {detected_count}/{len(pose_results)} frames detected")

        key_frames = select_key_frames(pose_results, n=settings.key_frame_count)
        angle_stats = compute_angle_stats(pose_results)

        # ── Stage 3: Gemini analysis ─────────────────────────────────────
        job_store.update_stage(job_id, PipelineStage.gemini_analysis, 40)
        _log("Sending to Gemini...")
        from app.services.video_service import frames_to_pil
        from app.services.gemini_service import analyze_climb

        key_rgb_frames = [r["annotated_frame"] for r in key_frames]
        key_pil_images = frames_to_pil(key_rgb_frames)
        feedback = await asyncio.to_thread(analyze_climb, key_pil_images, angle_stats)
        _log("Gemini analysis complete")

        # ── Stage 4: TTS ─────────────────────────────────────────────────
        job_store.update_stage(job_id, PipelineStage.tts_generation, 58)
        _log("Generating TTS...")
        from app.services.tts_service import synthesize

        audio_path = os.path.join(output_dir, "coaching_audio.mp3")
        await asyncio.to_thread(synthesize, feedback.overall_summary, audio_path)
        _log("TTS complete")

        # ── Stage 5: Assemble annotated video ────────────────────────────
        job_store.update_stage(job_id, PipelineStage.assembling_output, 75)
        _log("Assembling video...")
        from app.services.video_service import assemble_annotated_video

        annotated_video_path = os.path.join(output_dir, "annotated_climb.mp4")
        await asyncio.to_thread(assemble_annotated_video, pose_results, annotated_video_path, output_fps)
        _log("Video assembled")

        # ── Complete ─────────────────────────────────────────────────────
        output_urls = OutputUrls(
            annotated_video=f"/outputs/{job_id}/annotated_climb.mp4",
            coaching_audio=f"/outputs/{job_id}/coaching_audio.mp3",
        )
        job_store.complete(job_id, feedback, output_urls)
        _log("DONE")

    except Exception:
        error_msg = traceback.format_exc()
        print(f"[Pipeline] Job {job_id} failed:\n{error_msg}")
        job_store.fail(job_id, error_msg.splitlines()[-1])
