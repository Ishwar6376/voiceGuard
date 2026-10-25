"""
model_manager.py — ECAPA-TDNN speaker encoder via SpeechBrain

How speaker verification works here (short version):
  1. The pretrained ECAPA-TDNN network turns ~1.5 s of speech into a
     192-number "voiceprint" vector (an embedding).
  2. Recordings of the same person produce vectors that point in roughly the
     same direction; different people point in different directions.
  3. We compare two vectors with cosine similarity (1.0 = identical direction,
     0.0 = unrelated). Typical values for 1.5 s windows:
        same speaker       ~0.45 – 0.85
        different speaker  ~0.00 – 0.40

All embeddings returned from this module are L2-normalised (length 1), so
downstream code never has to worry about vector magnitude.

The network is never trained here — it was trained on thousands of speakers
(VoxCeleb) and is used as-is.
"""
import logging
import numpy as np
import torch
from pathlib import Path
from threading import Lock

logger = logging.getLogger("model_manager")

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_SOURCE    = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_CACHE_DIR = Path(__file__).parent / "storage" / "pretrained"
EMBEDDING_DIM   = 192
TARGET_SR       = 16000
MIN_AUDIO_S     = 0.5     # minimum audio duration for embedding extraction
MIN_RMS         = 0.001   # minimum RMS to consider audio non-silent

# Window size used both for live listening and for building reference
# voiceprints, so enrollment and live audio are compared like-for-like.
WINDOW_S = 1.5
HOP_S    = 0.75

# ─── Singleton encoder ───────────────────────────────────────────────────────
_lock       = Lock()
_encoder    = None
_load_error: str | None = None


def _load_encoder():
    """
    Lazily load the ECAPA-TDNN encoder from SpeechBrain (thread-safe).
    Downloads the model on first use (~80 MB) into storage/pretrained.
    """
    global _encoder, _load_error
    if _encoder is not None:
        return _encoder

    with _lock:
        if _encoder is not None:
            return _encoder

        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            _encoder = EncoderClassifier.from_hparams(
                source=MODEL_SOURCE,
                savedir=str(MODEL_CACHE_DIR),
                run_opts={"device": "cpu"},
                # Copy instead of symlink: symlinks need Developer Mode on Windows
                local_strategy=LocalStrategy.COPY,
            )
            _encoder.eval()
            _load_error = None
            logger.info("✅ ECAPA-TDNN speaker encoder loaded.")
        except Exception as e:
            _load_error = str(e)
            logger.error(f"❌ Failed to load ECAPA-TDNN encoder: {e}", exc_info=True)
            raise RuntimeError(
                f"Could not load ECAPA-TDNN encoder. "
                f"Ensure 'speechbrain' and 'torch' are installed. Error: {e}"
            ) from e

    return _encoder


# ─── Public API ──────────────────────────────────────────────────────────────

def get_embedding(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray | None:
    """
    Compute a unit-length 192-dim speaker embedding for one piece of audio.

    Returns None if the audio is too short, too quiet, or the embedding is
    degenerate.
    """
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    if len(audio) < TARGET_SR * MIN_AUDIO_S:
        return None

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < MIN_RMS:
        return None

    encoder = _load_encoder()
    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)

    with torch.no_grad():
        embedding = encoder.encode_batch(waveform)  # (1, 1, 192)

    embedding = embedding.squeeze().cpu().numpy().astype(np.float32)

    norm = float(np.linalg.norm(embedding))
    if norm < 1e-6 or not np.all(np.isfinite(embedding)):
        logger.warning(f"Degenerate embedding (norm={norm:.4f})")
        return None

    return embedding / norm


def get_window_embeddings(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Split audio into overlapping WINDOW_S windows and embed each one.

    Returns an (n_windows, 192) array of unit vectors (may be empty).
    Audio shorter than one window is embedded as a single piece.
    """
    from vad import is_speech

    win = int(WINDOW_S * sr)
    hop = int(HOP_S * sr)

    if len(audio) <= win:
        pieces = [audio]
    else:
        pieces = [audio[i:i + win] for i in range(0, len(audio) - win + 1, hop)]

    embeddings = []
    for piece in pieces:
        if not is_speech(piece, sr):
            continue
        emb = get_embedding(piece, sr)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return np.stack(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Raw cosine similarity in [-1, 1]. 1.0 = same direction."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def preload():
    """Load the encoder (call from a background thread at startup)."""
    try:
        _load_encoder()
    except RuntimeError:
        pass  # error already logged; surfaced via get_model_info()


def is_encoder_loaded() -> bool:
    return _encoder is not None


def get_model_info() -> dict:
    """Return model status information (never triggers a model load)."""
    return {
        "encoder_loaded": _encoder is not None,
        "encoder_error": _load_error,
        "encoder_type": "ECAPA-TDNN (SpeechBrain)",
        "embedding_dim": EMBEDDING_DIM,
        "target_sr": TARGET_SR,
        "window_s": WINDOW_S,
    }
