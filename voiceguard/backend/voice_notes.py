"""
voice_notes.py — Incremental voice note management for speaker learning

Voice notes are short recordings the user adds over time. For each note we
store the WAV plus the speaker embeddings of every 1.5 s window in it
(the same window size used when listening live).

  - The average of all window embeddings is the speaker's "centroid"
    voiceprint used for live matching.
  - With several notes, calibrator.py uses them to learn a personal threshold.
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

MIN_DURATION_S = 1.5    # at least one full analysis window
MAX_DURATION_S = 30.0
MIN_RMS        = 0.001
EMBED_VERSION  = 2      # 2 = per-window, unit-normalised embeddings


def _ensure_dir(speaker_id: str) -> Path:
    d = VOICE_NOTES_DIR / speaker_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── CRUD Operations ────────────────────────────────────────────────────────

def add_voice_note(speaker_id: str, audio_bytes: bytes, metadata: dict = None) -> dict:
    """
    Decode + validate audio, compute window embeddings, and store the note.

    Raises:
        ValueError: If audio is too short, too long, too quiet, or has no speech.
    """
    from model_manager import get_window_embeddings

    # ── 1. Load and validate audio ────────────────────────────────────────
    audio = load_audio(audio_bytes, target_sr=SR)
    duration = len(audio) / SR

    if duration < MIN_DURATION_S:
        raise ValueError(
            f"Voice note too short ({duration:.1f}s). Minimum is {MIN_DURATION_S}s."
        )
    if duration > MAX_DURATION_S:
        raise ValueError(
            f"Voice note too long ({duration:.1f}s). Maximum is {MAX_DURATION_S}s."
        )

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < MIN_RMS:
        raise ValueError("Voice note is too quiet. Please speak louder.")

    # ── 2. Compute embeddings ─────────────────────────────────────────────
    embeddings = get_window_embeddings(audio, SR)
    if len(embeddings) == 0:
        raise ValueError(
            "Not enough clear speech in this recording. Please talk continuously "
            "for the whole recording (e.g. read a sentence aloud), a little louder."
        )

    # ── 3. Save files ─────────────────────────────────────────────────────
    note_dir = _ensure_dir(speaker_id)
    note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    (note_dir / f"{note_id}.wav").write_bytes(audio_bytes)
    np.save(note_dir / f"{note_id}_embed.npy", embeddings)

    note_meta = {
        "note_id": note_id,
        "speaker_id": speaker_id,
        "recorded_at": datetime.now().isoformat(),
        "duration_s": round(duration, 2),
        "rms": round(rms, 4),
        "n_windows": int(len(embeddings)),
        "embed_version": EMBED_VERSION,
        **(metadata or {}),
    }
    with open(note_dir / f"{note_id}_meta.json", "w") as f:
        json.dump(note_meta, f, indent=2)

    logger.info(
        f"✅ Voice note '{note_id}' added for '{speaker_id}' "
        f"({duration:.1f}s, {len(embeddings)} windows)"
    )
    return note_meta


def list_voice_notes(speaker_id: str) -> list[dict]:
    """List all voice notes for a speaker, oldest first."""
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


# ─── Embedding Operations ───────────────────────────────────────────────────

def _load_note_embeddings(embed_file: Path) -> np.ndarray | None:
    """Load a note's embeddings as an (n, 192) array of unit vectors."""
    try:
        emb = np.load(embed_file).astype(np.float32)
    except Exception as e:
        logger.warning(f"Skipping corrupted embedding: {embed_file}: {e}")
        return None
    if emb.ndim == 1:
        emb = emb[None, :]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def get_note_embeddings(speaker_id: str) -> list[np.ndarray]:
    """One (n_windows, 192) array per voice note, oldest first."""
    note_dir = VOICE_NOTES_DIR / speaker_id
    if not note_dir.exists():
        return []
    result = []
    for embed_file in sorted(note_dir.glob("*_embed.npy")):
        emb = _load_note_embeddings(embed_file)
        if emb is not None and len(emb) > 0:
            result.append(emb)
    return result


def centroid_of(embeddings: np.ndarray) -> np.ndarray:
    """Unit-length mean of a stack of unit embeddings."""
    c = embeddings.mean(axis=0).astype(np.float32)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def get_speaker_centroid(speaker_id: str) -> np.ndarray | None:
    """
    The speaker's reference voiceprint: average of all window embeddings from
    all voice notes (unit length). None if the speaker has no notes.
    """
    notes = get_note_embeddings(speaker_id)
    if not notes:
        return None
    return centroid_of(np.concatenate(notes))


def upgrade_legacy_notes() -> int:
    """
    Re-compute embeddings for notes saved by older versions (one un-normalised
    embedding of the whole clip) so they use the current per-window format.
    Needs the encoder; call after it is loaded. Returns number upgraded.
    """
    from model_manager import get_window_embeddings

    if not VOICE_NOTES_DIR.exists():
        return 0

    upgraded = 0
    for meta_file in VOICE_NOTES_DIR.glob("*/*_meta.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("embed_version") == EMBED_VERSION:
                continue

            note_id = meta["note_id"]
            wav_path = meta_file.parent / f"{note_id}.wav"
            audio = load_audio(wav_path.read_bytes(), target_sr=SR)
            embeddings = get_window_embeddings(audio, SR)
            if len(embeddings) == 0:
                logger.warning(f"Could not upgrade '{note_id}': no speech found.")
                continue

            np.save(meta_file.parent / f"{note_id}_embed.npy", embeddings)
            meta.pop("embedding_norm", None)
            meta["n_windows"] = int(len(embeddings))
            meta["embed_version"] = EMBED_VERSION
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)
            upgraded += 1
            logger.info(f"⬆️  Upgraded embeddings for '{note_id}'")
        except Exception as e:
            logger.error(f"Failed to upgrade {meta_file}: {e}")
    return upgraded
