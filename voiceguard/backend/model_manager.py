"""
model_manager.py — ECAPA-TDNN speaker encoder via SpeechBrain
Replaces the legacy Resemblyzer + MFCC/DTW pipeline with a modern,
fine-tunable speaker embedding model.

Architecture:
  - Base encoder: ECAPA-TDNN (pre-trained on VoxCeleb1+2)
  - Embedding: 192-dimensional speaker vector
  - Inference: ~50ms per 1.5s window on CPU

The encoder is lazily loaded on first use and cached for the process lifetime.
All inference runs on CPU by default for simplicity and portability.
"""
import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from threading import Lock

logger = logging.getLogger("model_manager")

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_CACHE_DIR = Path(__file__).parent / "storage" / "pretrained"
EMBEDDING_DIM   = 192
TARGET_SR       = 16000
MIN_AUDIO_S     = 0.5     # minimum audio duration for embedding extraction
MIN_RMS         = 0.001   # minimum RMS to consider audio non-silent

# ─── Singleton encoder ───────────────────────────────────────────────────────
_lock    = Lock()
_encoder = None


def _load_encoder():
    """
    Lazily load the ECAPA-TDNN encoder from SpeechBrain.
    Thread-safe: uses a lock to prevent double-loading.
    Downloads the model on first use (~25MB).
    """
    global _encoder
    if _encoder is not None:
        return _encoder

    with _lock:
        # Double-check after acquiring lock
        if _encoder is not None:
            return _encoder

        try:
            from speechbrain.inference.speaker import EncoderClassifier

            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            _encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(MODEL_CACHE_DIR),
                run_opts={"device": "cpu"},
            )
            logger.info("✅ ECAPA-TDNN speaker encoder loaded from SpeechBrain.")
        except Exception as e:
            logger.error(f"❌ Failed to load ECAPA-TDNN encoder: {e}", exc_info=True)
            raise RuntimeError(
                f"Could not load ECAPA-TDNN encoder. "
                f"Ensure 'speechbrain' and 'torch' are installed. Error: {e}"
            ) from e

    return _encoder


# ─── Public API ──────────────────────────────────────────────────────────────

def get_embedding(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray | None:
    """
    Compute a 192-dim speaker embedding using ECAPA-TDNN.

    Args:
        audio: Float32 mono audio array.
        sr:    Sample rate of the input audio.

    Returns:
        192-dim float32 numpy embedding, or None if audio is
        too short, too quiet, or produces a degenerate embedding.
    """
    # ── Validate audio quality ────────────────────────────────────────────
    if len(audio) < sr * MIN_AUDIO_S:
        logger.debug(f"Audio too short ({len(audio)/sr:.2f}s < {MIN_AUDIO_S}s)")
        return None

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < MIN_RMS:
        logger.debug(f"Audio too quiet (RMS={rms:.5f})")
        return None

    # ── Load encoder ──────────────────────────────────────────────────────
    encoder = _load_encoder()

    # ── Prepare tensor ────────────────────────────────────────────────────
    waveform = torch.from_numpy(audio.copy()).float().unsqueeze(0)  # (1, time)

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SR
        )

    # ── Extract embedding ─────────────────────────────────────────────────
    with torch.no_grad():
        embedding = encoder.encode_batch(waveform)  # (1, 1, 192)

    embedding = embedding.squeeze().cpu().numpy().astype(np.float32)  # (192,)

    # ── Validate embedding ────────────────────────────────────────────────
    norm = float(np.linalg.norm(embedding))
    if norm < 0.1 or np.any(np.isnan(embedding)):
        logger.warning(f"Degenerate embedding (norm={norm:.4f})")
        return None

    logger.debug(f"Embedding OK — norm={norm:.4f}, rms={rms:.4f}")
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors, mapped to [0, 1].

    Raw cosine similarity ∈ [-1, 1] is mapped via (cos + 1) / 2.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    return (cos_sim + 1.0) / 2.0


def is_encoder_available() -> bool:
    """Check if the ECAPA-TDNN encoder can be loaded."""
    try:
        _load_encoder()
        return True
    except Exception:
        return False


def get_model_info() -> dict:
    """Return model status information."""
    return {
        "encoder_loaded": _encoder is not None,
        "encoder_type": "ECAPA-TDNN (SpeechBrain)",
        "embedding_dim": EMBEDDING_DIM,
        "target_sr": TARGET_SR,
        "model_cache": str(MODEL_CACHE_DIR),
    }
