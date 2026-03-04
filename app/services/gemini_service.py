"""
Gemini multimodal API integration using the google-genai SDK.
Sends key frames + angle stats and returns structured coaching feedback.
"""
import json
import re
from google import genai
from PIL import Image
from app.config import settings
from app.models.schemas import FeedbackResult, KeyMoment

_FALLBACK_FEEDBACK = FeedbackResult(
    overall_summary="Great effort on the wall! Keep focusing on your footwork and hip placement.",
    form=["Maintain straight arms when possible to conserve energy."],
    movement=["Try to move fluidly between holds rather than static movements."],
    route_reading=["Study the route from the ground before starting."],
    key_moments=[KeyMoment(frame_index=0, observation="Starting position noted.")],
    encouragement="You're making great progress -- keep climbing!",
)

_PROMPT_TEMPLATE = """You are an elite-level rock climbing technique analyst. You have coached V10+ boulderers and 5.13+ sport climbers. You are reviewing video frames and joint angle telemetry from a climber's session.

FRAMES: I am providing {n_frames} sequential key frames from the climb.
TELEMETRY: Joint angle measurements computed via pose estimation:
{angle_stats}

YOUR TASK: Provide specific, actionable, technical feedback based on what you ACTUALLY SEE in these frames and the telemetry. Do NOT give generic climbing advice. Every point must reference something observable — a body position, a visible movement pattern, or a specific frame.

IMPORTANT: Use plain, conversational language. Do NOT cite degree values or numbers from the telemetry. Instead, use the data to inform your observations and describe what you see in natural terms like "your arms are significantly bent," "your hips are pulling away from the wall," "your legs are nearly straight," etc. The telemetry helps you understand what's happening — the climber just needs to hear it described clearly.

ANALYSIS CRITERIA — evaluate each:

1. ARM TENSION: Are the arms staying bent when they could be straight? On holds at or below shoulder height, hanging on straight arms saves energy. Look for signs of over-gripping or muscling through moves.

2. HIP POSITION: Are hips rotated into the wall (flagging, drop-knee, twist-lock) or squared off and sagging away? Hips far from the wall means wasted energy and poor balance.

3. FOOT PRECISION: Can you see deliberate foot placements or sloppy/rushed feet? Are they using the toe tip (precise) or smearing the whole foot? Is the climber looking at feet before placing?

4. CENTER OF GRAVITY: Is weight centered over feet or hanging from arms? On steep terrain, deep knee bends help. On slabs, standing tall over feet is key.

5. MOVEMENT EFFICIENCY: Static vs dynamic movement. Are they deadpointing? Cutting feet unnecessarily? Moving smoothly between holds or pausing/readjusting?

6. REST POSITIONS: Do you see any shaking out, chalking, or deliberate rest positions? Arms that stay bent the entire climb suggest the climber isn't resting enough.

For each feedback point, be specific about what you see: "In frame 3, your left arm is noticeably bent while gripping a hold at chest height — try straightening that arm to hang on your skeleton rather than your muscles" rather than "keep your arms straight."

Return ONLY valid JSON with NO markdown fences:
{{
  "overall_summary": "2-3 sentences summarizing the most important technical observations in plain language. This will be read aloud as coaching audio.",
  "form": ["specific observation referencing what you see in the frames", "..."],
  "movement": ["specific observation about movement patterns seen in frames", "..."],
  "route_reading": ["specific observation about sequencing, body positioning choices, rest usage", "..."],
  "key_moments": [{{"frame_index": 0, "observation": "specific technical observation about this frame"}}],
  "encouragement": "one sentence acknowledging something specific they did well"
}}"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_response(text: str) -> FeedbackResult:
    try:
        cleaned = _strip_json_fences(text)
        data = json.loads(cleaned)
        key_moments = [KeyMoment(**km) for km in data.get("key_moments", [])]
        return FeedbackResult(
            overall_summary=data.get("overall_summary", ""),
            form=data.get("form", []),
            movement=data.get("movement", []),
            route_reading=data.get("route_reading", []),
            key_moments=key_moments,
            encouragement=data.get("encouragement", ""),
        )
    except Exception as exc:
        print(f"[GeminiService] JSON parse failed: {exc}")
        print(f"[GeminiService] Raw response: {text[:500]}")
        return _FALLBACK_FEEDBACK


def analyze_climb(key_frames_pil: list[Image.Image], angle_stats: dict) -> FeedbackResult:
    """
    Send key frames + angle stats to Gemini and return FeedbackResult.
    """
    if not settings.gemini_api_key:
        print("[GeminiService] No API key set, using fallback")
        return _FALLBACK_FEEDBACK

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        # Format angle stats
        stats_lines = []
        for joint, stat in angle_stats.items():
            if joint == "detection_rate":
                stats_lines.append(f"  pose_detection_rate: {stat:.1%}")
            elif isinstance(stat, dict):
                stats_lines.append(
                    f"  {joint}: mean={stat['mean']:.1f}, "
                    f"min={stat['min']:.1f}, max={stat['max']:.1f}, "
                    f"std={stat['std']:.1f}"
                )
        angle_stats_str = "\n".join(stats_lines) if stats_lines else "  (no pose detected)"

        prompt = _PROMPT_TEMPLATE.format(
            n_frames=len(key_frames_pil),
            angle_stats=angle_stats_str,
        )

        # Build content: prompt text + images
        contents = [prompt] + list(key_frames_pil)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        result = _parse_response(response.text)

        if not result.overall_summary:
            print("[GeminiService] Empty summary, retrying once")
            response2 = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            result = _parse_response(response2.text)
            if not result.overall_summary:
                return _FALLBACK_FEEDBACK

        return result

    except Exception as exc:
        print(f"[GeminiService] Error: {exc}")
        return _FALLBACK_FEEDBACK
