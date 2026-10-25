"""
audio_processor.py — Audio loading
"""
import io

import librosa
import numpy as np
import soundfile as sf

SR = 16000


def load_audio(data: bytes, target_sr: int = SR) -> np.ndarray:
    """
    Load audio bytes → mono float32 array at target_sr.
    Tries three formats in order:
      1. soundfile  — fast, handles WAV/FLAC/OGG-Vorbis
      2. librosa    — handles more formats via audioread (needs ffmpeg for WebM)
      3. raw numpy  — last resort if bytes are already raw float32 PCM
    """
    buf = io.BytesIO(data)

    # ── Attempt 1: soundfile ─────────────────────────────────────────────
    try:
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
    except Exception:
        pass

    # ── Attempt 2: librosa (uses audioread / ffmpeg under the hood) ──────
    try:
        buf.seek(0)
        audio, sr = librosa.load(buf, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception:
        pass

    # ── Attempt 3: raw float32 PCM ───────────────────────────────────────
    if len(data) % 4 == 0 and len(data) > 0:
        audio = np.frombuffer(data, dtype=np.float32)
        if np.all(np.isfinite(audio)) and np.max(np.abs(audio)) <= 10:
            return audio.copy()

    raise ValueError(
        "Could not decode audio. "
        "Ensure the uploaded file is WAV, FLAC, OGG, or raw PCM float32. "
        "For WebM/Opus, ffmpeg must be installed on the server."
    )
