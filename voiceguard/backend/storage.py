"""
storage.py — Speaker profiles and cascading deletion

Storage layout:
  storage/
  ├── config.json                       # door settings (challenge phrase, unlock time)
  ├── access_log.jsonl                  # every granted / denied attempt
  ├── templates/{speaker_id}_meta.json  # speaker profile (name, threshold)
  ├── voice_notes/{speaker_id}/         # voice note recordings + embeddings
  │   ├── note_xxx.wav
  │   ├── note_xxx_embed.npy
  │   └── note_xxx_meta.json
  ├── models/{speaker_id}/              # calibration results
  │   └── calibration.json
  ├── pretrained/                       # SpeechBrain model cache (downloaded)
  └── raw/                              # legacy v1 enrollment WAVs (migration)
"""
import json
import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger("storage")

STORAGE_DIR     = Path(__file__).parent / "storage"
TEMPLATE_DIR    = STORAGE_DIR / "templates"
RAW_DIR         = STORAGE_DIR / "raw"
VOICE_NOTES_DIR = STORAGE_DIR / "voice_notes"
MODELS_DIR      = STORAGE_DIR / "models"

CONFIG_PATH     = STORAGE_DIR / "config.json"

SPEAKER_ID_RE = re.compile(r"^[a-z0-9_]{1,48}$")

DEFAULT_CONFIG = {
    "require_challenge": True,  # speaker must say the random phrase shown on screen
    "unlock_seconds": 5,        # door re-locks automatically after this long
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULT_CONFIG})
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save_config(cfg: dict):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def ensure_dirs():
    for d in [TEMPLATE_DIR, VOICE_NOTES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def make_speaker_id(name: str) -> str:
    """'Ishwar K-2' → 'ishwar_k_2' (safe for filenames)."""
    sid = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return sid[:48]


def is_valid_speaker_id(speaker_id: str) -> bool:
    return bool(SPEAKER_ID_RE.match(speaker_id or ""))


# ─── Speaker profile ─────────────────────────────────────────────────────────

def save_speaker_profile(speaker_id: str, meta: dict):
    ensure_dirs()
    with open(TEMPLATE_DIR / f"{speaker_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_speaker_profile(speaker_id: str) -> dict | None:
    if not is_valid_speaker_id(speaker_id):
        return None
    meta_path = TEMPLATE_DIR / f"{speaker_id}_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


# ─── Speaker listing ────────────────────────────────────────────────────────

def list_speakers() -> list[dict]:
    """List all enrolled speakers with voice note counts and calibration status."""
    from calibrator import DEFAULT_THRESHOLD, get_calibration
    from voice_notes import get_voice_note_count

    ensure_dirs()
    speakers = []

    for meta_file in sorted(TEMPLATE_DIR.glob("*_meta.json")):
        speaker_id = meta_file.name[: -len("_meta.json")]
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        calibration = get_calibration(speaker_id)
        speakers.append({
            "id": speaker_id,
            "name": meta.get("name", speaker_id),
            "enrolled_at": meta.get("enrolled_at", ""),
            "voice_note_count": get_voice_note_count(speaker_id),
            "calibrated": calibration is not None,
            "calibrated_at": calibration["calibrated_at"] if calibration else None,
            "similarity_threshold": meta.get("similarity_threshold", DEFAULT_THRESHOLD),
        })

    return speakers


# ─── Speaker deletion (cascading) ───────────────────────────────────────────

def delete_speaker(speaker_id: str) -> bool:
    """Delete a speaker profile, voice notes, calibration and legacy files."""
    if not is_valid_speaker_id(speaker_id):
        return False

    deleted = False

    for suffix in ["_mfcc.npy", "_embed.npy", "_meta.json"]:
        f = TEMPLATE_DIR / f"{speaker_id}{suffix}"
        if f.exists():
            f.unlink()
            deleted = True

    raw = RAW_DIR / f"{speaker_id}_enrollment.wav"
    if raw.exists():
        raw.unlink()
        deleted = True

    for d in [VOICE_NOTES_DIR / speaker_id, MODELS_DIR / speaker_id]:
        if d.exists():
            shutil.rmtree(d)
            deleted = True

    if deleted:
        logger.info(f"🗑️ Deleted all data for speaker '{speaker_id}'")
    return deleted
