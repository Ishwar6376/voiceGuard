"""
voice_notes.py — Incremental voice note management for speaker learning

Voice notes are short audio recordings that users add over time.
Each note is stored as a WAV file alongside its pre-computed speaker embedding.
The embeddings are averaged into a "centroid" for similarity-based matching,
and used as training data for fine-tuning the classifier head.
"""
import uuid
import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from audio_processor import load_audio, SR

logger = logging.getLogger("voice_notes")

# ─── Configuration ───────────────────────────────────────────────────────────
VOICE_NOTES_DIR = Path(__file__).parent / "storage" / "voice_notes"

MIN_DURATION_S = 1.0    # minimum voice note duration in seconds
MAX_DURATION_S = 30.0   # maximum voice note duration in seconds
MIN_RMS        = 0.001  # minimum RMS to consider audio non-silent
MIN_NOTES_FOR_FINETUNE = 5


def _ensure_dir(speaker_id: str) -> Path:
    """Ensure the voice notes directory exists for a speaker."""
    d = VOICE_NOTES_DIR / speaker_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── CRUD Operations ────────────────────────────────────────────────────────

def add_voice_note(speaker_id: str, audio_bytes: bytes, metadata: dict = None) -> dict:
    """
    Add a voice note for a speaker.

    Steps:
      1. Decode and validate the audio
      2. Compute the ECAPA-TDNN embedding
      3. Store WAV + embedding + metadata

    Args:
        speaker_id:  Enrolled speaker identifier.
        audio_bytes: Raw audio file bytes (WAV, FLAC, etc.).
        metadata:    Optional extra metadata to store with the note.

    Returns:
        Note metadata dict: {note_id, speaker_id, duration_s, rms, embedding_norm, ...}

    Raises:
        ValueError: If audio is too short, too long, too quiet, or embedding fails.
    """
    # Lazy import to avoid circular dependency at module load time
    from model_manager import get_embedding

    note_dir = _ensure_dir(speaker_id)

    # ── 1. Load and validate audio ────────────────────────────────────────
    audio = load_audio(audio_bytes, target_sr=SR)
    duration = len(audio) / SR

    if duration < MIN_DURATION_S:
        raise ValueError(
            f"Voice note too short ({duration:.1f}s). "
            f"Minimum is {MIN_DURATION_S}s."
        )
    if duration > MAX_DURATION_S:
        raise ValueError(
            f"Voice note too long ({duration:.1f}s). "
            f"Maximum is {MAX_DURATION_S}s."
        )

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < MIN_RMS:
        raise ValueError("Voice note is too quiet. Please speak louder.")

    # ── 2. Compute embedding ──────────────────────────────────────────────
    embedding = get_embedding(audio, SR)
    if embedding is None:
        raise ValueError(
            "Could not extract a speaker embedding from this recording. "
            "Please try recording again with clearer speech."
        )

    # ── 3. Generate unique note ID ────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    note_id = f"note_{timestamp}_{short_uuid}"

    # ── 4. Save files ─────────────────────────────────────────────────────
    # WAV audio
    wav_path = note_dir / f"{note_id}.wav"
    wav_path.write_bytes(audio_bytes)

    # Cached embedding
    embed_path = note_dir / f"{note_id}_embed.npy"
    np.save(embed_path, embedding)

    # Metadata
    note_meta = {
        "note_id": note_id,
        "speaker_id": speaker_id,
        "recorded_at": datetime.now().isoformat(),
        "duration_s": round(duration, 2),
        "rms": round(rms, 4),
        "embedding_norm": round(float(np.linalg.norm(embedding)), 4),
        **(metadata or {}),
    }

    meta_path = note_dir / f"{note_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(note_meta, f, indent=2)

    logger.info(
        f"✅ Voice note '{note_id}' added for speaker '{speaker_id}' "
        f"({duration:.1f}s, emb_norm={note_meta['embedding_norm']:.3f})"
    )
    return note_meta


def list_voice_notes(speaker_id: str) -> list[dict]:
    """
    List all voice notes for a speaker, sorted by recording date.
    Returns a list of metadata dicts.
    """
    note_dir = VOICE_NOTES_DIR / speaker_id
    if not note_dir.exists():
        return []

    notes = []
    for meta_file in sorted(note_dir.glob("*_meta.json")):
        try:
            with open(meta_file) as f:
                notes.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping corrupted note metadata: {meta_file}: {e}")

    return notes


def get_voice_note_count(speaker_id: str) -> int:
    """Count voice notes for a speaker."""
    note_dir = VOICE_NOTES_DIR / speaker_id
    if not note_dir.exists():
        return 0
    return len(list(note_dir.glob("*_meta.json")))


def delete_voice_note(speaker_id: str, note_id: str) -> bool:
    """Delete a single voice note and its associated files."""
    note_dir = VOICE_NOTES_DIR / speaker_id
    deleted = False
    for suffix in [".wav", "_embed.npy", "_meta.json"]:
        f = note_dir / f"{note_id}{suffix}"
        if f.exists():
            f.unlink()
            deleted = True

    if deleted:
        logger.info(f"🗑️ Deleted voice note '{note_id}' for speaker '{speaker_id}'")
    return deleted


def delete_all_voice_notes(speaker_id: str) -> int:
    """Delete all voice notes for a speaker. Returns count deleted."""
    note_dir = VOICE_NOTES_DIR / speaker_id
    if not note_dir.exists():
        return 0

    count = 0
    for f in note_dir.iterdir():
        if f.is_file():
            f.unlink()
            count += 1

    # Remove the empty directory
    try:
        note_dir.rmdir()
    except OSError:
        pass

    logger.info(f"🗑️ Deleted {count} files for speaker '{speaker_id}'")
    return count


# ─── Embedding Operations ───────────────────────────────────────────────────

def get_speaker_centroid(speaker_id: str) -> np.ndarray | None:
    """
    Compute the centroid (L2-normalized average) embedding from all voice notes.

    Used for similarity-based matching before a fine-tuned classifier is available.
    Returns None if no embeddings are stored.
    """
    note_dir = VOICE_NOTES_DIR / speaker_id
    if not note_dir.exists():
        return None

    embeddings = []
    for embed_file in sorted(note_dir.glob("*_embed.npy")):
        try:
            emb = np.load(embed_file)
            embeddings.append(emb)
        except Exception as e:
            logger.warning(f"Skipping corrupted embedding: {embed_file}: {e}")

    if not embeddings:
        return None

    # Average and L2-normalize
    centroid = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    return centroid


def get_all_embeddings(speaker_id: str) -> list[np.ndarray]:
    """Load all cached embeddings for a speaker, sorted by filename."""
    note_dir = VOICE_NOTES_DIR / speaker_id
    if not note_dir.exists():
        return []

    embeddings = []
    for embed_file in sorted(note_dir.glob("*_embed.npy")):
        try:
            embeddings.append(np.load(embed_file))
        except Exception as e:
            logger.warning(f"Skipping corrupted embedding: {embed_file}: {e}")

    return embeddings


def can_fine_tune(speaker_id: str) -> bool:
    """Check if a speaker has enough voice notes for fine-tuning."""
    return get_voice_note_count(speaker_id) >= MIN_NOTES_FOR_FINETUNE
