"""
migrate.py — One-time migration from VoiceGuard v1 → v2

Scans legacy storage/raw/ for existing enrollment WAV files and
re-processes each as the first voice note for the corresponding speaker.

Usage:
    python migrate.py
"""
import logging
import sys
from pathlib import Path

# Ensure backend directory is on path
sys.path.insert(0, str(Path(__file__).parent))

from voice_notes import add_voice_note, get_voice_note_count
from calibrator import DEFAULT_THRESHOLD
from storage import TEMPLATE_DIR, RAW_DIR, load_speaker_profile, save_speaker_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("migrate")


def migrate():
    """
    Migrate all legacy enrolled speakers to the v2 voice notes system.

    For each speaker found in storage/templates/*_meta.json:
      1. Check if they already have voice notes (skip if so)
      2. Find their legacy enrollment audio in storage/raw/
      3. Re-process it as the first voice note (computes ECAPA-TDNN embedding)
      4. Update speaker profile with v2 threshold fields
    """
    if not TEMPLATE_DIR.exists():
        logger.info("No legacy templates directory found. Nothing to migrate.")
        return

    meta_files = list(TEMPLATE_DIR.glob("*_meta.json"))
    if not meta_files:
        logger.info("No legacy speaker metadata found. Nothing to migrate.")
        return

    logger.info(f"Found {len(meta_files)} legacy speaker(s) to migrate.")

    migrated = 0
    skipped = 0
    failed = 0

    for meta_file in meta_files:
        speaker_id = meta_file.stem.replace("_meta", "")

        # Skip if already has voice notes
        if get_voice_note_count(speaker_id) > 0:
            logger.info(f"⏭️  '{speaker_id}' already has voice notes. Skipping.")
            skipped += 1
            continue

        # Find legacy enrollment audio
        raw_path = RAW_DIR / f"{speaker_id}_enrollment.wav"
        if not raw_path.exists():
            logger.warning(f"⚠️  No enrollment audio found for '{speaker_id}'. Skipping.")
            skipped += 1
            continue

        try:
            # Read the legacy enrollment WAV
            audio_bytes = raw_path.read_bytes()

            # Add as first voice note (this computes the ECAPA-TDNN embedding)
            note_meta = add_voice_note(
                speaker_id,
                audio_bytes,
                metadata={"source": "migration_from_v1"},
            )

            # Update speaker profile with v2 fields
            profile = load_speaker_profile(speaker_id)
            if profile:
                profile["similarity_threshold"] = DEFAULT_THRESHOLD
                profile["score_version"] = 2
                save_speaker_profile(speaker_id, profile)

            logger.info(
                f"✅ Migrated '{speaker_id}' → voice note '{note_meta['note_id']}' "
                f"(duration: {note_meta['duration_s']}s)"
            )
            migrated += 1

        except Exception as e:
            logger.error(f"❌ Failed to migrate '{speaker_id}': {e}")
            failed += 1

    logger.info(
        f"\n{'='*60}\n"
        f"Migration complete!\n"
        f"  ✅ Migrated: {migrated}\n"
        f"  ⏭️  Skipped:  {skipped}\n"
        f"  ❌ Failed:   {failed}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    migrate()
