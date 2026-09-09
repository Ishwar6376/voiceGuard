"""
audio_processor.py — Audio loading and feature extraction

Retained for audio I/O and optional MFCC visualization.
DTW distance matching has been removed (replaced by DL embeddings).
"""
import numpy as np
import librosa
import soundfile as sf
import io
from typing import Optional
from vad import trim_silence

# ─── Config ─────────────────────────────────────────────────────────────────
SR        = 16000
N_MFCC    = 13
N_FFT     = 512
HOP_LEN   = 160   # 10ms
WIN_LEN   = 400   # 25ms
N_MELS    = 40
PREEMPH   = 0.97


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
        return audio
    except Exception as e1:
        pass  # fall through

    # ── Attempt 2: librosa (uses audioread / ffmpeg under the hood) ──────
    try:
        buf.seek(0)
        audio, sr = librosa.load(buf, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception as e2:
        pass  # fall through

    # ── Attempt 3: raw float32 PCM (from WebSocket streaming) ───────────
    try:
        audio = np.frombuffer(data, dtype=np.float32)
        if len(audio) > 0:
            return audio
    except Exception as e3:
        pass

    raise ValueError(
        "Could not decode audio. "
        "Ensure the uploaded file is WAV, FLAC, OGG, or raw PCM float32. "
        "For WebM/Opus, ffmpeg must be installed on the server."
    )


def extract_mfcc(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract MFCC features from audio.
    Returns (T, 39) matrix: 13 MFCCs + 13 deltas + 13 delta-deltas.
    Retained for feature visualization; no longer used for matching.
    """
    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - PREEMPH * audio[:-1])

    # Trim silence
    audio = trim_silence(audio, sr)

    if len(audio) < WIN_LEN:
        audio = np.pad(audio, (0, WIN_LEN - len(audio)))

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        n_mels=N_MELS,
        fmin=80, fmax=8000,
        window="hamming"
    )

    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])  # (39, T)
    return features.T  # (T, 39)
