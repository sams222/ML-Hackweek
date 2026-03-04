"""
ElevenLabs TTS integration with MOCK_TTS dev flag support.
"""
import os
import shutil
from app.config import settings

# Path to a bundled silent/placeholder MP3 used in mock mode
_MOCK_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "../../assets/mock_audio.mp3")


def synthesize(text: str, output_path: str) -> str:
    """
    Convert text to speech and write MP3 to output_path.
    Returns output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if settings.mock_tts:
        _write_mock_audio(output_path)
        return output_path

    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        audio = client.text_to_speech.convert(
            voice_id=settings.elevenlabs_voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        return output_path

    except Exception as exc:
        print(f"[TTSService] Error: {exc}, falling back to mock audio")
        _write_mock_audio(output_path)
        return output_path


def _write_mock_audio(output_path: str) -> None:
    """Write a minimal silent MP3 file for dev/testing."""
    mock_src = os.path.abspath(_MOCK_AUDIO_PATH)
    if os.path.exists(mock_src):
        shutil.copy2(mock_src, output_path)
    else:
        # Write a minimal valid MP3 header (silent, 1 frame)
        silent_mp3 = _minimal_silent_mp3()
        with open(output_path, "wb") as f:
            f.write(silent_mp3)


def _minimal_silent_mp3() -> bytes:
    """Return bytes of a minimal silent MP3 (ID3 tag + one silent MPEG frame)."""
    # ID3v2 header (empty)
    id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00"
    # One silent MPEG1 Layer3 frame at 128kbps, 44100Hz, stereo
    # Frame sync + header bytes for silence
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
    return id3 + frame
