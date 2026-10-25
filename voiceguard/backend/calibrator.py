"""
calibrator.py — Learn a personal similarity threshold from voice notes

Why not train a neural network head?
  With only a handful of recordings of ONE person and no recordings of other
  people, a trained classifier has nothing real to learn "not this person"
  from — it ends up accepting almost any sound. The pretrained ECAPA encoder
  already separates speakers well; what varies per person/microphone/room is
  how high the similarity score for the genuine speaker typically is.

What calibration does:
  1. Leave-one-out: for each voice note, build a voiceprint from the OTHER
     notes and score this note's 1.5 s windows against it. These are
     realistic "genuine speaker" scores (the note was never seen).
  2. If other speakers are enrolled, score their windows against this
     speaker's voiceprint — these are real "impostor" scores.
  3. Pick a threshold just below most genuine scores, but never below the
     impostor scores (plus a safety margin) or a global floor.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from voice_notes import VOICE_NOTES_DIR, get_note_embeddings, centroid_of

logger = logging.getLogger("calibrator")

CALIBRATION_DIR = Path(__file__).parent / "storage" / "models"

MIN_NOTES_FOR_CALIBRATION = 3

DEFAULT_THRESHOLD = 0.50     # used until the speaker is calibrated
THRESHOLD_FLOOR   = 0.45     # never go below: rejects noise and most other voices
THRESHOLD_CEIL    = 0.80     # never go above: would reject the genuine speaker
GENUINE_PERCENTILE = 10      # aim to accept ~90% of genuine windows
IMPOSTOR_MARGIN   = 0.05


def _calibration_path(speaker_id: str) -> Path:
    return CALIBRATION_DIR / speaker_id / "calibration.json"


def can_calibrate(note_count: int) -> bool:
    return note_count >= MIN_NOTES_FOR_CALIBRATION


def get_calibration(speaker_id: str) -> dict | None:
    path = _calibration_path(speaker_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def delete_calibration(speaker_id: str):
    path = _calibration_path(speaker_id)
    if path.exists():
        path.unlink()
    try:
        path.parent.rmdir()
    except OSError:
        pass


def calibrate(speaker_id: str) -> dict:
    """
    Compute and store a personal threshold. Raises ValueError if there are
    not enough usable voice notes.
    """
    notes = get_note_embeddings(speaker_id)
    if len(notes) < MIN_NOTES_FOR_CALIBRATION:
        raise ValueError(
            f"Need at least {MIN_NOTES_FOR_CALIBRATION} voice notes to calibrate "
            f"(have {len(notes)})."
        )

    # ── 1. Genuine scores (leave-one-note-out) ────────────────────────────
    genuine = []
    for i, held_out in enumerate(notes):
        others = np.concatenate([n for j, n in enumerate(notes) if j != i])
        genuine.extend((held_out @ centroid_of(others)).tolist())
    genuine = np.array(genuine)

    # ── 2. Impostor scores (other enrolled speakers, if any) ──────────────
    centroid = centroid_of(np.concatenate(notes))
    impostor = []
    if VOICE_NOTES_DIR.exists():
        for other_dir in VOICE_NOTES_DIR.iterdir():
            if other_dir.is_dir() and other_dir.name != speaker_id:
                for emb in get_note_embeddings(other_dir.name):
                    impostor.extend((emb @ centroid).tolist())
    impostor = np.array(impostor)

    # ── 3. Choose threshold ───────────────────────────────────────────────
    genuine_low = float(np.percentile(genuine, GENUINE_PERCENTILE))
    threshold = genuine_low
    warning = None

    if len(impostor) > 0:
        safe = float(np.max(impostor)) + IMPOSTOR_MARGIN
        if safe <= genuine_low:
            threshold = genuine_low
        else:
            # Voices overlap: pick the threshold with the fewest total mistakes
            # (wrongly rejected genuine windows + wrongly accepted impostor windows)
            candidates = np.unique(np.concatenate([genuine, impostor]))
            errors = [(genuine < t).mean() + (impostor >= t).mean() for t in candidates]
            threshold = float(candidates[int(np.argmin(errors))])
            warning = (
                "Another enrolled voice scores close to yours, so the two may "
                "sometimes be confused. Record more voice notes to separate them."
            )

    threshold = float(np.clip(threshold, THRESHOLD_FLOOR, THRESHOLD_CEIL))
    if genuine_low < THRESHOLD_FLOOR and warning is None:
        warning = (
            "Your voice notes are not very consistent with each other. Record "
            "a few more notes in the room/mic you'll use for monitoring."
        )

    result = {
        "speaker_id": speaker_id,
        "threshold": round(threshold, 4),
        "calibrated_at": datetime.now().isoformat(),
        "n_notes": len(notes),
        "n_genuine_windows": int(len(genuine)),
        "n_impostor_windows": int(len(impostor)),
        "genuine_mean": round(float(genuine.mean()), 4),
        "genuine_p10": round(genuine_low, 4),
        "impostor_max": round(float(impostor.max()), 4) if len(impostor) else None,
        # Share of held-out genuine windows that pass the threshold
        "genuine_accept_rate": round(float((genuine >= threshold).mean()), 4),
        "impostor_reject_rate": (
            round(float((impostor < threshold).mean()), 4) if len(impostor) else None
        ),
        "warning": warning,
    }

    path = _calibration_path(speaker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"🎯 Calibrated '{speaker_id}': threshold={threshold:.3f} "
        f"(genuine mean={result['genuine_mean']}, p10={result['genuine_p10']}, "
        f"impostor max={result['impostor_max']})"
    )
    return result
