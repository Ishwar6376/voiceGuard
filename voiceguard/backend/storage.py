"""
storage.py — Save and load speaker templates + voice notes + model checkpoints

Storage layout:
  storage/
  ├── voice_notes/{speaker_id}/       # incremental voice note recordings
  │   ├── note_xxx.wav
  │   ├── note_xxx_embed.npy
  │   └── note_xxx_meta.json
  ├── models/{speaker_id}/            # fine-tuned classifier heads
  │   ├── classifier_head.pt
  │   └── classifier_config.json
  ├── pretrained/                     # SpeechBrain model cache
  ├── templates/                      # legacy MFCC/embed templates (migration)
  └── raw/                            # legacy enrollment WAVs (migration)
"""
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger("storage")

STORAGE_DIR    = Path(__file__).parent / "storage"
TEMPLATE_DIR   = STORAGE_DIR / "templates"
RAW_DIR        = STORAGE_DIR / "raw"
VOICE_NOTES_DIR = STORAGE_DIR / "voice_notes"
MODELS_DIR     = STORAGE_DIR / "models"


def ensure_dirs():
    """Create all required storage directories."""
    for d in [TEMPLATE_DIR, RAW_DIR, VOICE_NOTES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─── Speaker metadata (slim profile — voice notes are stored separately) ────

def save_speaker_profile(speaker_id: str, meta: dict):
    """Save or update a speaker's profile metadata."""
    ensure_dirs()
    meta_path = TEMPLATE_DIR / f"{speaker_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_speaker_profile(speaker_id: str) -> dict | None:
    """Load a speaker's profile metadata. Returns None if not found."""
    meta_path = TEMPLATE_DIR / f"{speaker_id}_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


# ─── Legacy template support (for migration compatibility) ──────────────────

def save_template(
    speaker_id: str,
    mfcc_matrix: np.ndarray,
    embedding: Optional[np.ndarray],
    raw_audio_bytes: bytes,
    meta: dict,
):
    """Legacy: Save a speaker template (MFCC + embedding + raw audio + metadata)."""
    ensure_dirs()
    np.save(TEMPLATE_DIR / f"{speaker_id}_mfcc.npy", mfcc_matrix)

    if embedding is not None:
        np.save(TEMPLATE_DIR / f"{speaker_id}_embed.npy", embedding)

    raw_path = RAW_DIR / f"{speaker_id}_enrollment.wav"
    raw_path.write_bytes(raw_audio_bytes)

    meta_path = TEMPLATE_DIR / f"{speaker_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_template(speaker_id: str) -> dict:
    """Legacy: Load a speaker template."""
    ensure_dirs()
    result = {"speaker_id": speaker_id}

    mfcc_path = TEMPLATE_DIR / f"{speaker_id}_mfcc.npy"
    result["mfcc"] = np.load(mfcc_path) if mfcc_path.exists() else None

    embed_path = TEMPLATE_DIR / f"{speaker_id}_embed.npy"
    result["embedding"] = np.load(embed_path) if embed_path.exists() else None

    meta_path = TEMPLATE_DIR / f"{speaker_id}_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            result["meta"] = json.load(f)
    else:
        result["meta"] = {}

    return result


# ─── Speaker listing ────────────────────────────────────────────────────────

def list_speakers() -> list[dict]:
    """
    List all enrolled speakers with their metadata, voice note counts,
    and fine-tuning status.
    """
    ensure_dirs()
    speakers = []

    for meta_file in TEMPLATE_DIR.glob("*_meta.json"):
        speaker_id = meta_file.stem.replace("_meta", "")
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Count voice notes
        notes_dir = VOICE_NOTES_DIR / speaker_id
        note_count = len(list(notes_dir.glob("*_meta.json"))) if notes_dir.exists() else 0

        # Check for fine-tuned model
        model_dir = MODELS_DIR / speaker_id
        has_classifier = (model_dir / "classifier_head.pt").exists()

        # Load classifier config for training info
        classifier_config = None
        config_path = model_dir / "classifier_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    classifier_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        speakers.append({
            "id": speaker_id,
            "name": meta.get("name", speaker_id),
            "enrolled_at": meta.get("enrolled_at", ""),
            "keyword": meta.get("keyword", ""),
            "voice_note_count": note_count,
            "has_classifier": has_classifier,
            "classifier_accuracy": (
                classifier_config.get("final_accuracy")
                if classifier_config else None
            ),
            "classifier_trained_at": (
                classifier_config.get("trained_at")
                if classifier_config else None
            ),
            # Legacy fields for backward compat
            "has_dl_embedding": (TEMPLATE_DIR / f"{speaker_id}_embed.npy").exists(),
            "dtw_threshold": meta.get("dtw_threshold", 0.5),
            "dl_threshold": meta.get("dl_threshold", 0.75),
            "fusion_threshold": meta.get("fusion_threshold", 0.45),
            # New v2 thresholds
            "similarity_threshold": meta.get("similarity_threshold", 0.65),
            "classifier_threshold": meta.get("classifier_threshold", 0.70),
        })

    return speakers


# ─── Speaker deletion (cascading) ───────────────────────────────────────────

def delete_speaker(speaker_id: str) -> bool:
    """
    Delete a speaker and ALL associated data:
    - Legacy templates (MFCC, embed, meta)
    - Legacy raw enrollment audio
    - Voice notes
    - Fine-tuned model checkpoints
    """
    ensure_dirs()
    deleted = False

    # 1. Legacy template files
    for suffix in ["_mfcc.npy", "_embed.npy", "_meta.json"]:
        f = TEMPLATE_DIR / f"{speaker_id}{suffix}"
        if f.exists():
            f.unlink()
            deleted = True

    # 2. Legacy raw audio
    raw = RAW_DIR / f"{speaker_id}_enrollment.wav"
    if raw.exists():
        raw.unlink()
        deleted = True

    # 3. Voice notes
    notes_dir = VOICE_NOTES_DIR / speaker_id
    if notes_dir.exists():
        for f in notes_dir.iterdir():
            if f.is_file():
                f.unlink()
                deleted = True
        try:
            notes_dir.rmdir()
        except OSError:
            pass

    # 4. Model checkpoints
    model_dir = MODELS_DIR / speaker_id
    if model_dir.exists():
        for f in model_dir.iterdir():
            if f.is_file():
                f.unlink()
                deleted = True
        try:
            model_dir.rmdir()
        except OSError:
            pass

    if deleted:
        logger.info(f"🗑️ Deleted all data for speaker '{speaker_id}'")

    return deleted
