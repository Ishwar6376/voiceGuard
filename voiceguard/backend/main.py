"""
main.py — VoiceGuard voice-lock FastAPI backend

Endpoints:
  REST:
    - POST   /enroll                                 — Enroll a new speaker (creates first voice note)
    - GET    /speakers                               — List all speakers
    - DELETE /speakers/{id}                          — Delete a speaker + all data
    - POST   /speakers/{id}/voice-notes              — Upload a voice note (form field: file)
    - GET    /speakers/{id}/voice-notes              — List voice notes
    - DELETE /speakers/{id}/voice-notes/{note_id}    — Delete a voice note
    - POST   /speakers/{id}/calibrate                — Learn a personal threshold from voice notes
    - GET    /speakers/{id}/model-status             — Voice note count + calibration status
    - GET    /speakers/{id}/spectrum                 — FFT spectrum of latest voice note
    - POST   /settings/{id}                          — Update a speaker's similarity threshold
    - GET    /config, POST /config                   — Door settings (challenge phrase, unlock time)
    - GET    /lock/status, POST /lock/lock           — Door state / lock immediately
    - GET    /access-log                             — Recent granted/denied attempts
    - GET    /model/status, GET /health

  WebSocket /listen — live audio → speaker identification → door lock
    client → server:
      {"type": "start"}
      {"type": "audio_chunk", "data": <base64 float32 PCM>, "sample_rate": 48000}
      {"type": "transcript", "text": "..."}          (browser speech-to-text)
      {"type": "stop"}
    server → client:
      status | challenge | phrase_heard | level | match | lock | access | error | ping
"""
import asyncio
import base64
import json
import logging
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soxr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from audio_processor import SR
from calibrator import (
    DEFAULT_THRESHOLD, MIN_NOTES_FOR_CALIBRATION,
    calibrate, can_calibrate, get_calibration,
)
from challenge import new_phrase, phrase_spoken
from lock_manager import door
from model_manager import get_embedding, get_model_info, preload
from storage import (
    TEMPLATE_DIR, delete_speaker, list_speakers, load_config,
    load_speaker_profile, make_speaker_id, save_config, save_speaker_profile,
)
from vad import is_speech
from voice_notes import (
    VOICE_NOTES_DIR, add_voice_note, delete_voice_note, get_speaker_centroid,
    get_voice_note_count, list_voice_notes, upgrade_legacy_notes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("voiceguard")

# ─── Live verification config ────────────────────────────────────────────────
WINDOW_DURATION  = 1.5   # seconds of audio per embedding
OVERLAP_DURATION = 0.75  # seconds shared between consecutive windows
CHUNK_SR         = 16000

# The door opens when the SAME enrolled speaker wins MATCHES_REQUIRED of the
# last DECISION_WINDOWS speech windows (all within DECISION_MAX_AGE_S).
# One lucky window is not enough, which removes most false accepts.
DECISION_WINDOWS   = 3
MATCHES_REQUIRED   = 2
DECISION_MAX_AGE_S = 5.0

# Challenge phrase timing. Speech-to-text reports the phrase ~1 s after it was
# spoken, so windows from just before are checked too (they're still "recent").
PHRASE_VALID_S  = 5.0    # heard phrase counts this long
ATTEMPT_WAIT_S  = 3.0    # after the phrase, wait this long for voice match → else DENIED
CHALLENGE_TTL_S = 60.0   # unused phrases are replaced after this long

NOTE_ID_RE = re.compile(r"^note_[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$")


# ─── Startup ─────────────────────────────────────────────────────────────────

def _upgrade_legacy_profiles():
    """Profiles from older versions stored thresholds on a different scale."""
    for meta_file in TEMPLATE_DIR.glob("*_meta.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("score_version") == 2:
                continue
            for key in ["dtw_threshold", "dl_threshold", "fusion_threshold", "classifier_threshold"]:
                meta.pop(key, None)
            meta["similarity_threshold"] = DEFAULT_THRESHOLD
            meta["score_version"] = 2
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"⬆️  Upgraded profile {meta_file.name} (threshold reset to {DEFAULT_THRESHOLD})")
        except Exception as e:
            logger.error(f"Failed to upgrade profile {meta_file}: {e}")


def _warmup():
    """Load the model and upgrade old data without blocking the server."""
    preload()
    _upgrade_legacy_profiles()
    if get_model_info()["encoder_loaded"]:
        upgrade_legacy_notes()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warmup, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(title="VoiceGuard API", version="3.0.0", lifespan=lifespan)

# Only pages served from this laptop may call the API
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_speaker(speaker_id: str) -> dict:
    profile = load_speaker_profile(speaker_id)
    if profile is None:
        raise HTTPException(404, f"Speaker '{speaker_id}' not enrolled.")
    return profile


# ══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "encoder_loaded": get_model_info()["encoder_loaded"],
        "model_info": get_model_info(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model/status")
async def model_status():
    return get_model_info()


# ─── Enrollment ──────────────────────────────────────────────────────────────

@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):
    """
    Enroll a new speaker. The uploaded audio becomes the speaker's first
    voice note; more can be added via /speakers/{id}/voice-notes.
    """
    speaker_id = make_speaker_id(name)
    if not speaker_id:
        raise HTTPException(400, "Please use a name containing letters or numbers.")
    if load_speaker_profile(speaker_id) is not None:
        raise HTTPException(
            409,
            f"'{name}' is already enrolled. Add voice notes to improve it, "
            f"or delete the speaker first.",
        )

    raw_bytes = await file.read()

    # Validate + embed the audio before creating the profile
    try:
        note_meta = await asyncio.to_thread(
            add_voice_note, speaker_id, raw_bytes, {"source": "enrollment"}
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    meta = {
        "name": name.strip(),
        "speaker_id": speaker_id,
        "enrolled_at": datetime.now().isoformat(),
        "similarity_threshold": DEFAULT_THRESHOLD,
        "score_version": 2,
        "sample_rate": SR,
    }
    save_speaker_profile(speaker_id, meta)

    logger.info(f"✅ Enrolled '{name}' as '{speaker_id}' — first note {note_meta['note_id']}")
    return {
        "success": True,
        "speaker_id": speaker_id,
        "name": meta["name"],
        "first_voice_note": note_meta,
        "meta": meta,
    }


# ─── Speaker management ─────────────────────────────────────────────────────

@app.get("/speakers")
async def get_speakers():
    return {"speakers": list_speakers()}


@app.delete("/speakers/{speaker_id}")
async def remove_speaker(speaker_id: str):
    if not delete_speaker(speaker_id):
        raise HTTPException(404, f"Speaker '{speaker_id}' not found.")
    return {"success": True, "speaker_id": speaker_id}


# ─── Voice Notes ─────────────────────────────────────────────────────────────

@app.post("/speakers/{speaker_id}/voice-notes")
async def upload_voice_note(speaker_id: str, file: UploadFile = File(...)):
    _require_speaker(speaker_id)
    raw_bytes = await file.read()

    try:
        note_meta = await asyncio.to_thread(add_voice_note, speaker_id, raw_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))

    note_count = get_voice_note_count(speaker_id)
    return {
        "success": True,
        "voice_note": note_meta,
        "total_notes": note_count,
        "can_calibrate": can_calibrate(note_count),
    }


@app.get("/speakers/{speaker_id}/voice-notes")
async def get_voice_notes(speaker_id: str):
    _require_speaker(speaker_id)
    notes = list_voice_notes(speaker_id)
    return {
        "speaker_id": speaker_id,
        "voice_notes": notes,
        "total": len(notes),
        "can_calibrate": can_calibrate(len(notes)),
    }


@app.delete("/speakers/{speaker_id}/voice-notes/{note_id}")
async def remove_voice_note(speaker_id: str, note_id: str):
    _require_speaker(speaker_id)
    if not NOTE_ID_RE.match(note_id):
        raise HTTPException(400, "Invalid note id.")
    if get_voice_note_count(speaker_id) <= 1:
        raise HTTPException(400, "Cannot delete the last voice note. Delete the speaker instead.")
    if not delete_voice_note(speaker_id, note_id):
        raise HTTPException(404, f"Voice note '{note_id}' not found.")
    return {
        "success": True,
        "note_id": note_id,
        "remaining_notes": get_voice_note_count(speaker_id),
    }


# ─── Calibration ─────────────────────────────────────────────────────────────

@app.post("/speakers/{speaker_id}/calibrate")
async def trigger_calibration(speaker_id: str):
    """Learn a personal similarity threshold from the speaker's voice notes."""
    profile = _require_speaker(speaker_id)
    try:
        result = await asyncio.to_thread(calibrate, speaker_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

    profile["similarity_threshold"] = result["threshold"]
    save_speaker_profile(speaker_id, profile)
    return {"success": True, **result}


@app.get("/speakers/{speaker_id}/model-status")
async def speaker_model_status(speaker_id: str):
    profile = _require_speaker(speaker_id)
    note_count = get_voice_note_count(speaker_id)
    calibration = get_calibration(speaker_id)

    return {
        "speaker_id": speaker_id,
        "voice_note_count": note_count,
        "min_notes_for_calibration": MIN_NOTES_FOR_CALIBRATION,
        "can_calibrate": can_calibrate(note_count),
        "calibrated": calibration is not None,
        # Notes were added/removed since calibrating → should recalibrate
        "calibration_stale": bool(calibration and calibration["n_notes"] != note_count),
        "calibration": calibration,
        "similarity_threshold": profile.get("similarity_threshold", DEFAULT_THRESHOLD),
        "default_threshold": DEFAULT_THRESHOLD,
        "base_model": get_model_info(),
    }


# ─── Spectrum ────────────────────────────────────────────────────────────────

@app.get("/speakers/{speaker_id}/spectrum")
async def get_spectrum(speaker_id: str):
    """
    Time-averaged magnitude spectrum of the speaker's latest voice note.
    Returns 128 bins (normalized 0–1) for the frontend FFT overlay.
    """
    _require_speaker(speaker_id)

    notes = list_voice_notes(speaker_id)
    if not notes:
        raise HTTPException(404, "No audio found for this speaker.")

    note_path = VOICE_NOTES_DIR / speaker_id / f"{notes[-1]['note_id']}.wav"
    try:
        audio, _ = librosa.load(str(note_path), sr=16000, mono=True)
    except Exception as e:
        raise HTTPException(500, f"Could not load audio: {e}")

    mag = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    avg = mag.mean(axis=1)
    peak = float(avg.max())
    if peak > 0:
        avg = avg / peak

    TARGET = 128
    spectrum = avg[np.linspace(0, len(avg) - 1, TARGET).astype(int)]

    return {
        "speaker_id": speaker_id,
        "spectrum": [round(float(v), 4) for v in spectrum],
        "n_bins": TARGET,
        "sample_rate": 16000,
    }


# ─── Settings ────────────────────────────────────────────────────────────────

@app.post("/settings/{speaker_id}")
async def update_settings(speaker_id: str, settings: dict):
    profile = _require_speaker(speaker_id)

    if "similarity_threshold" in settings:
        try:
            value = float(settings["similarity_threshold"])
        except (TypeError, ValueError):
            raise HTTPException(400, "similarity_threshold must be a number.")
        if not 0.0 < value < 1.0:
            raise HTTPException(400, "similarity_threshold must be between 0 and 1.")
        profile["similarity_threshold"] = round(value, 4)

    save_speaker_profile(speaker_id, profile)
    return {"success": True, "updated": profile}


@app.get("/config")
async def get_config():
    return load_config()


@app.post("/config")
async def update_config(settings: dict):
    cfg = load_config()
    if "require_challenge" in settings:
        cfg["require_challenge"] = bool(settings["require_challenge"])
    if "unlock_seconds" in settings:
        try:
            seconds = float(settings["unlock_seconds"])
        except (TypeError, ValueError):
            raise HTTPException(400, "unlock_seconds must be a number.")
        if not 1 <= seconds <= 60:
            raise HTTPException(400, "unlock_seconds must be between 1 and 60.")
        cfg["unlock_seconds"] = seconds
    save_config(cfg)
    return cfg


# ─── Lock ────────────────────────────────────────────────────────────────────

@app.get("/lock/status")
async def lock_status():
    return door.status()


@app.post("/lock/lock")
async def lock_now():
    door.lock("manual")
    return door.status()


@app.get("/access-log")
async def access_log():
    return {"events": door.recent_events()}


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET: LIVE LISTENING → DOOR LOCK
# ══════════════════════════════════════════════════════════════════════════════

def _load_references() -> list[dict]:
    """Voiceprint + threshold of every enrolled speaker that has voice notes."""
    refs = []
    for s in list_speakers():
        centroid = get_speaker_centroid(s["id"])
        if centroid is not None:
            refs.append({
                "id": s["id"],
                "name": s["name"],
                "threshold": float(s["similarity_threshold"]),
                "calibrated": s["calibrated"],
                "centroid": centroid,
            })
    return refs


def _score_window(window: np.ndarray, centroids: np.ndarray) -> Optional[np.ndarray]:
    """Speech check + embedding + similarity to every speaker. Runs in a worker thread."""
    if not is_speech(window, CHUNK_SR):
        return None
    emb = get_embedding(window, CHUNK_SR)
    if emb is None:
        return None
    return centroids @ emb  # both unit length → cosine similarity per speaker


@app.websocket("/listen")
async def listen_ws(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected.")

    refs: list[dict] = []
    centroids: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None
    cfg = load_config()
    listening = False

    audio_buffer = np.array([], dtype=np.float32)
    resampler = None
    resampler_sr = None

    # Recent speech windows: (time, index of matched speaker or None, best score, best index)
    recent: deque = deque(maxlen=DECISION_WINDOWS)

    phrase: Optional[str] = None
    phrase_issued_at = 0.0
    phrase_heard_at: Optional[float] = None
    attempt_deadline: Optional[float] = None

    window_samples = int(WINDOW_DURATION * CHUNK_SR)
    step_samples = window_samples - int(OVERLAP_DURATION * CHUNK_SR)

    async def send(msg: dict):
        await ws.send_text(json.dumps(msg))

    async def on_door_event(msg: dict):
        try:
            await send(msg)
        except Exception:
            pass

    async def new_challenge():
        nonlocal phrase, phrase_issued_at, phrase_heard_at, attempt_deadline
        phrase = new_phrase() if cfg["require_challenge"] else None
        phrase_issued_at = time.monotonic()
        phrase_heard_at = None
        attempt_deadline = None
        await send({"type": "challenge", "phrase": phrase, "expires_in": CHALLENGE_TTL_S})

    def voice_leader() -> tuple[Optional[int], int, float]:
        """Speaker who matched most recent windows: (index, n_matches, best score)."""
        now = time.monotonic()
        counts: dict[int, int] = {}
        best: dict[int, float] = {}
        for t, idx, score, _ in recent:
            if idx is not None and now - t <= DECISION_MAX_AGE_S:
                counts[idx] = counts.get(idx, 0) + 1
                best[idx] = max(best.get(idx, 0.0), score)
        if not counts:
            return None, 0, 0.0
        idx = max(counts, key=lambda i: (counts[i], best[i]))
        return idx, counts[idx], best[idx]

    def phrase_ok() -> bool:
        if not cfg["require_challenge"]:
            return True
        return phrase_heard_at is not None and time.monotonic() - phrase_heard_at <= PHRASE_VALID_S

    async def evaluate():
        """Unlock, deny, or keep waiting."""
        if not door.is_locked:
            return

        idx, n_match, score = voice_leader()
        if idx is not None and n_match >= MATCHES_REQUIRED and phrase_ok():
            door.unlock(refs[idx]["id"], refs[idx]["name"], score, cfg["unlock_seconds"])
            recent.clear()
            if cfg["require_challenge"]:
                await new_challenge()
            return

        if attempt_deadline is not None and time.monotonic() > attempt_deadline:
            # Phrase was spoken, but the voice didn't match anyone enrolled
            now = time.monotonic()
            scored = [(s, b) for t, _, s, b in recent if now - t <= DECISION_MAX_AGE_S]
            if scored:
                s, b = max(scored)
                door.record("denied", reason="voice not recognised",
                            closest=refs[b]["name"], score=s,
                            threshold=refs[b]["threshold"])
            else:
                door.record("denied", reason="no clear speech heard")
            recent.clear()
            await new_challenge()

    door.add_listener(on_door_event)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await send({"type": "ping"})
                continue

            try:
                msg = json.loads(raw)
                msg_type = msg["type"]
            except (json.JSONDecodeError, KeyError, TypeError):
                await send({"type": "error", "message": "Malformed message."})
                continue

            # ── START ────────────────────────────────────────────────────
            if msg_type == "start":
                refs = await asyncio.to_thread(_load_references)
                if not refs:
                    await send({"type": "error", "message": "No speakers enrolled yet. Enroll someone first."})
                    continue

                centroids = np.stack([r["centroid"] for r in refs])
                thresholds = np.array([r["threshold"] for r in refs])
                cfg = load_config()
                recent.clear()
                audio_buffer = np.array([], dtype=np.float32)
                resampler = None
                listening = True

                await send({
                    "type": "status",
                    "listening": True,
                    "speakers": [
                        {k: r[k] for k in ("id", "name", "threshold", "calibrated")} for r in refs
                    ],
                    "require_challenge": cfg["require_challenge"],
                    "unlock_seconds": cfg["unlock_seconds"],
                    "matches_required": MATCHES_REQUIRED,
                    "decision_windows": DECISION_WINDOWS,
                })
                await send({"type": "lock", **door.status()})
                await new_challenge()
                logger.info(
                    f"Listening for {len(refs)} speaker(s): {[r['name'] for r in refs]}, "
                    f"challenge phrase {'ON' if cfg['require_challenge'] else 'OFF'}"
                )

            # ── SPEECH-TO-TEXT FROM BROWSER ──────────────────────────────
            elif msg_type == "transcript":
                if (listening and phrase and phrase_heard_at is None and door.is_locked
                        and phrase_spoken(str(msg.get("text", "")), phrase)):
                    phrase_heard_at = time.monotonic()
                    attempt_deadline = phrase_heard_at + ATTEMPT_WAIT_S
                    logger.info(f"Challenge phrase heard: '{msg.get('text')}'")
                    await send({"type": "phrase_heard", "phrase": phrase})
                    await evaluate()

            # ── STOP ─────────────────────────────────────────────────────
            elif msg_type == "stop":
                listening = False
                recent.clear()
                audio_buffer = np.array([], dtype=np.float32)
                await send({"type": "status", "listening": False})

            # ── AUDIO CHUNK ──────────────────────────────────────────────
            elif msg_type == "audio_chunk" and listening:
                chunk = np.frombuffer(base64.b64decode(msg["data"]), dtype=np.float32)

                client_sr = int(msg.get("sample_rate", CHUNK_SR))
                if client_sr != CHUNK_SR:
                    # Streaming resampler keeps state between chunks (no clicks at edges)
                    if resampler is None or resampler_sr != client_sr:
                        resampler = soxr.ResampleStream(client_sr, CHUNK_SR, 1, dtype="float32")
                        resampler_sr = client_sr
                    chunk = resampler.resample_chunk(chunk)

                audio_buffer = np.concatenate([audio_buffer, chunk])
                # Never let the backlog grow unbounded if processing falls behind
                if len(audio_buffer) > 4 * window_samples:
                    audio_buffer = audio_buffer[-2 * window_samples:]

                rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
                await send({
                    "type": "level",
                    "rms": round(rms, 4),
                    "voice": bool(len(chunk) and is_speech(chunk, CHUNK_SR, min_voiced_s=0)),
                })

                while len(audio_buffer) >= window_samples:
                    window = audio_buffer[:window_samples]
                    audio_buffer = audio_buffer[step_samples:]

                    scores = await asyncio.to_thread(_score_window, window, centroids)
                    if scores is None:
                        continue  # silence / noise — not scored

                    best = int(np.argmax(scores))
                    matched = bool(scores[best] >= thresholds[best])
                    recent.append((time.monotonic(), best if matched else None, float(scores[best]), best))
                    leader, n_match, _ = voice_leader()

                    await send({
                        "type": "match",
                        "best_speaker": refs[best]["id"],
                        "best_name": refs[best]["name"],
                        "similarity_score": round(float(scores[best]), 4),
                        "threshold": round(float(thresholds[best]), 4),
                        "matched": matched,
                        "scores": {r["name"]: round(float(s), 4) for r, s in zip(refs, scores)},
                        "leader": refs[leader]["name"] if leader is not None else None,
                        "recent_matches": n_match,
                        "matches_required": MATCHES_REQUIRED,
                        "phrase_required": cfg["require_challenge"],
                        "phrase_ok": phrase_ok(),
                        "timestamp": datetime.now().isoformat(),
                    })

                    await evaluate()

                # Time-based checks (run on every chunk, ~10× per second)
                if attempt_deadline is not None:
                    await evaluate()
                elif phrase and time.monotonic() - phrase_issued_at > CHALLENGE_TTL_S:
                    await new_challenge()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        door.remove_listener(on_door_event)


# ══════════════════════════════════════════════════════════════════════════════
# SERVE FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1 = reachable only from this laptop, so nobody else on the
    # Wi-Fi can enroll their voice or unlock the door.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
