"""
main.py — VoiceGuard v2 FastAPI Backend

Endpoints:
  REST:
    - POST   /enroll                                 — Enroll a new speaker (creates first voice note)
    - GET    /speakers                               — List all speakers
    - DELETE /speakers/{id}                           — Delete a speaker + all data
    - POST   /speakers/{id}/voice-notes              — Upload a voice note
    - GET    /speakers/{id}/voice-notes              — List voice notes
    - DELETE /speakers/{id}/voice-notes/{note_id}    — Delete a voice note
    - POST   /speakers/{id}/fine-tune                — Trigger fine-tuning
    - GET    /speakers/{id}/model-status             — Get model/classifier status
    - GET    /speakers/{id}/spectrum                 — FFT spectrum of latest voice note
    - POST   /settings/{id}                          — Update thresholds
    - GET    /model/status                           — Base model health
    - GET    /health                                 — Backend health
    - GET    /alarm/status                           — Alarm status
    - POST   /alarm/stop                             — Stop alarm

  WebSocket:
    - /listen  — Live audio streaming + speaker verification
"""
import asyncio
import base64
import json
import logging
import time
import io
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
import librosa

from fastapi import (
    FastAPI, File, Form, UploadFile,
    HTTPException, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from audio_processor import load_audio, SR
from model_manager import get_embedding, cosine_similarity, is_encoder_available, get_model_info
from alarm_manager import alarm
from storage import (
    save_speaker_profile, load_speaker_profile,
    list_speakers, delete_speaker,
    save_template, load_template,
)
from voice_notes import (
    add_voice_note, list_voice_notes, get_voice_note_count,
    delete_voice_note, delete_all_voice_notes,
    get_speaker_centroid, can_fine_tune,
)
from fine_tuner import fine_tuner
from vad import is_voice_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("voiceguard")

app = FastAPI(title="VoiceGuard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Sliding window config ───────────────────────────────────────────────────
WINDOW_DURATION  = 1.5   # seconds
OVERLAP_DURATION = 0.75  # seconds
CHUNK_SR         = 16000

# Default thresholds
DEFAULT_SIMILARITY_THRESHOLD  = 0.65   # cosine similarity (0-1) for centroid matching
DEFAULT_CLASSIFIER_THRESHOLD  = 0.70   # classifier confidence for fine-tuned matching


# ══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dl_encoder": is_encoder_available(),
        "model_info": get_model_info(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model/status")
async def model_status():
    """Get detailed model status."""
    return get_model_info()


# ─── Enrollment ──────────────────────────────────────────────────────────────

@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    file: UploadFile = File(...),
    keyword: str = Form(default=""),
):
    """
    Enroll a new speaker from an uploaded WAV file.

    The uploaded audio becomes the speaker's FIRST voice note.
    Additional voice notes can be added later via /speakers/{id}/voice-notes.
    """
    raw_bytes = await file.read()

    try:
        audio = load_audio(raw_bytes, target_sr=SR)
    except Exception as e:
        raise HTTPException(400, f"Could not read audio file: {e}")

    if len(audio) < SR * 0.5:
        raise HTTPException(400, "Recording too short. Please record at least 0.5 seconds.")

    # Generate speaker ID from name
    speaker_id = name.lower().replace(" ", "_").replace("-", "_")

    # Save speaker profile metadata
    meta = {
        "name": name,
        "speaker_id": speaker_id,
        "keyword": keyword.strip().lower() if keyword else "",
        "enrolled_at": datetime.now().isoformat(),
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "classifier_threshold": DEFAULT_CLASSIFIER_THRESHOLD,
        "sample_rate": SR,
        # Legacy fields for backward compat
        "dtw_threshold": 0.12,
        "dl_threshold": 0.72,
        "fusion_threshold": 0.45,
    }

    save_speaker_profile(speaker_id, meta)

    # Add the enrollment audio as the FIRST voice note
    try:
        note_meta = add_voice_note(
            speaker_id,
            raw_bytes,
            metadata={"source": "enrollment"},
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    logger.info(
        f"✅ Enrolled speaker '{name}' as '{speaker_id}' — "
        f"first voice note: {note_meta['note_id']}"
    )

    return {
        "success": True,
        "speaker_id": speaker_id,
        "name": name,
        "first_voice_note": note_meta,
        "dl_embedding": True,
        "meta": meta,
    }


# ─── Speaker management ─────────────────────────────────────────────────────

@app.get("/speakers")
async def get_speakers():
    return {"speakers": list_speakers()}


@app.delete("/speakers/{speaker_id}")
async def remove_speaker(speaker_id: str):
    # Also remove fine-tuner cache
    fine_tuner.delete_classifier(speaker_id)
    deleted = delete_speaker(speaker_id)
    if not deleted:
        raise HTTPException(404, f"Speaker '{speaker_id}' not found.")
    return {"success": True, "speaker_id": speaker_id}


# ─── Voice Notes ─────────────────────────────────────────────────────────────

@app.post("/speakers/{speaker_id}/voice-notes")
async def upload_voice_note(speaker_id: str, file: UploadFile = File(...)):
    """Upload a new voice note for a speaker."""
    # Verify speaker exists
    profile = load_speaker_profile(speaker_id)
    if profile is None:
        raise HTTPException(404, f"Speaker '{speaker_id}' not enrolled.")

    raw_bytes = await file.read()

    try:
        note_meta = add_voice_note(speaker_id, raw_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Return updated counts
    note_count = get_voice_note_count(speaker_id)
    can_ft = can_fine_tune(speaker_id)

    return {
        "success": True,
        "voice_note": note_meta,
        "total_notes": note_count,
        "can_fine_tune": can_ft,
    }


@app.get("/speakers/{speaker_id}/voice-notes")
async def get_voice_notes(speaker_id: str):
    """List all voice notes for a speaker."""
    profile = load_speaker_profile(speaker_id)
    if profile is None:
        raise HTTPException(404, f"Speaker '{speaker_id}' not enrolled.")

    notes = list_voice_notes(speaker_id)
    return {
        "speaker_id": speaker_id,
        "voice_notes": notes,
        "total": len(notes),
        "can_fine_tune": can_fine_tune(speaker_id),
    }


@app.delete("/speakers/{speaker_id}/voice-notes/{note_id}")
async def remove_voice_note(speaker_id: str, note_id: str):
    """Delete a single voice note."""
    deleted = delete_voice_note(speaker_id, note_id)
    if not deleted:
        raise HTTPException(404, f"Voice note '{note_id}' not found.")
    return {
        "success": True,
        "note_id": note_id,
        "remaining_notes": get_voice_note_count(speaker_id),
    }


# ─── Fine-Tuning ────────────────────────────────────────────────────────────

@app.post("/speakers/{speaker_id}/fine-tune")
async def trigger_fine_tune(speaker_id: str):
    """Trigger fine-tuning of the classification head for a speaker."""
    profile = load_speaker_profile(speaker_id)
    if profile is None:
        raise HTTPException(404, f"Speaker '{speaker_id}' not enrolled.")

    if fine_tuner.is_training:
        raise HTTPException(409, "Fine-tuning is already in progress. Please wait.")

    if not can_fine_tune(speaker_id):
        note_count = get_voice_note_count(speaker_id)
        raise HTTPException(
            400,
            f"Need at least 5 voice notes to fine-tune. Currently have {note_count}.",
        )

    # Run fine-tuning in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, fine_tuner.run_fine_tune, speaker_id
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}", exc_info=True)
        raise HTTPException(500, f"Fine-tuning failed: {e}")

    return result


@app.get("/speakers/{speaker_id}/model-status")
async def speaker_model_status(speaker_id: str):
    """Get model/classifier status for a specific speaker."""
    profile = load_speaker_profile(speaker_id)
    if profile is None:
        raise HTTPException(404, f"Speaker '{speaker_id}' not enrolled.")

    note_count = get_voice_note_count(speaker_id)
    has_clf = fine_tuner.has_classifier(speaker_id)
    clf_config = fine_tuner.get_config(speaker_id)

    return {
        "speaker_id": speaker_id,
        "voice_note_count": note_count,
        "can_fine_tune": can_fine_tune(speaker_id),
        "has_classifier": has_clf,
        "classifier_config": clf_config,
        "matching_mode": "classifier" if has_clf else "similarity",
        "base_model": get_model_info(),
    }


# ─── Spectrum ────────────────────────────────────────────────────────────────

@app.get("/speakers/{speaker_id}/spectrum")
async def get_spectrum(speaker_id: str):
    """
    Compute the time-averaged magnitude spectrum of the speaker's voice.
    Uses the latest voice note, or falls back to legacy enrollment audio.
    Returns 128 bins (normalized 0–1) for the frontend FFT overlay.
    """
    from pathlib import Path

    # Try latest voice note first
    audio = None
    notes = list_voice_notes(speaker_id)
    if notes:
        latest_note = notes[-1]
        note_path = (
            Path(__file__).parent / "storage" / "voice_notes"
            / speaker_id / f"{latest_note['note_id']}.wav"
        )
        if note_path.exists():
            try:
                audio, sr = librosa.load(str(note_path), sr=16000, mono=True)
            except Exception:
                pass

    # Fallback to legacy enrollment
    if audio is None or len(audio) == 0:
        raw_path = Path(__file__).parent / "storage" / "raw" / f"{speaker_id}_enrollment.wav"
        if raw_path.exists():
            try:
                audio, sr = librosa.load(str(raw_path), sr=16000, mono=True)
            except Exception as e:
                raise HTTPException(500, f"Could not load audio: {e}")

    if audio is None or len(audio) == 0:
        raise HTTPException(404, "No audio found for this speaker.")

    # STFT → average magnitude
    n_fft = 1024
    hop_length = 256
    mag = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    avg = mag.mean(axis=1)

    peak = float(avg.max())
    if peak > 0:
        avg = avg / peak

    # Downsample to 128 bins
    TARGET = 128
    indices = np.linspace(0, len(avg) - 1, TARGET).astype(int)
    spectrum = avg[indices]

    return {
        "speaker_id": speaker_id,
        "spectrum": [round(float(v), 4) for v in spectrum],
        "n_bins": TARGET,
        "sample_rate": 16000,
    }


# ─── Alarm ───────────────────────────────────────────────────────────────────

@app.get("/alarm/status")
async def alarm_status():
    return alarm.status()


@app.post("/alarm/stop")
async def stop_alarm():
    alarm.stop()
    return {"success": True, "message": "Alarm stopped."}


# ─── Settings ────────────────────────────────────────────────────────────────

@app.post("/settings/{speaker_id}")
async def update_settings(speaker_id: str, settings: dict):
    """Update thresholds for a speaker."""
    profile = load_speaker_profile(speaker_id)
    if profile is None:
        raise HTTPException(404, "Speaker not found.")

    # Update allowed keys
    allowed = [
        "similarity_threshold", "classifier_threshold",
        "dtw_threshold", "dl_threshold", "fusion_threshold",
    ]
    for key in allowed:
        if key in settings:
            profile[key] = float(settings[key])

    save_speaker_profile(speaker_id, profile)
    return {"success": True, "updated": profile}


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET: LIVE LISTENING
# ══════════════════════════════════════════════════════════════════════════════

active_connections: list[WebSocket] = []


@app.websocket("/listen")
async def listen_ws(ws: WebSocket):
    await ws.accept()
    active_connections.append(ws)
    logger.info("WebSocket connected.")

    # Per-connection state
    audio_buffer = np.array([], dtype=np.float32)
    speaker_id: Optional[str] = None
    speaker_profile: Optional[dict] = None
    centroid: Optional[np.ndarray] = None
    listening = False

    # Text filter gate
    text_gate_open = False
    keyword: str = ""

    window_samples = int(WINDOW_DURATION * CHUNK_SR)
    overlap_samples = int(OVERLAP_DURATION * CHUNK_SR)
    step_samples = window_samples - overlap_samples

    # Alarm listener for this connection
    async def send_alarm(msg: dict):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            pass

    alarm.add_listener(send_alarm)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))
                continue

            msg = json.loads(raw)

            # ── START ────────────────────────────────────────────────────
            if msg["type"] == "start":
                speaker_id = msg.get("speaker_id")

                # Load speaker profile
                speaker_profile = load_speaker_profile(speaker_id)
                if speaker_profile is None:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"Speaker '{speaker_id}' not enrolled.",
                    }))
                    continue

                # Load centroid embedding for similarity matching
                centroid = get_speaker_centroid(speaker_id)
                if centroid is None:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"No voice notes found for '{speaker_id}'. Please add voice notes first.",
                    }))
                    continue

                # Keyword setup
                keyword = speaker_profile.get("keyword", "").strip().lower()
                if msg.get("keyword"):
                    keyword = msg["keyword"].strip().lower()

                text_gate_open = False
                listening = True
                audio_buffer = np.array([], dtype=np.float32)

                # Determine matching mode
                has_classifier = fine_tuner.has_classifier(speaker_id)
                matching_mode = "classifier" if has_classifier else "similarity"
                note_count = get_voice_note_count(speaker_id)

                await ws.send_text(json.dumps({
                    "type": "status",
                    "listening": True,
                    "speaker": speaker_id,
                    "keyword": keyword,
                    "matching_mode": matching_mode,
                    "voice_note_count": note_count,
                    "has_classifier": has_classifier,
                }))
                logger.info(
                    f"Listening for speaker: {speaker_id}, "
                    f"keyword: '{keyword}', mode: {matching_mode}"
                )

            # ── TEXT MATCH ───────────────────────────────────────────────
            elif msg["type"] == "text_match":
                matched_keyword = msg.get("matched", False)
                transcript = msg.get("transcript", "")
                text_gate_open = bool(matched_keyword)
                logger.debug(
                    f"Text gate {'OPEN' if text_gate_open else 'CLOSED'} "
                    f"— transcript: '{transcript}'"
                )
                await ws.send_text(json.dumps({
                    "type": "text_gate",
                    "open": text_gate_open,
                    "transcript": transcript,
                }))

            # ── STOP ─────────────────────────────────────────────────────
            elif msg["type"] == "stop":
                listening = False
                text_gate_open = False
                audio_buffer = np.array([], dtype=np.float32)
                await ws.send_text(json.dumps({"type": "status", "listening": False}))

            # ── AUDIO CHUNK ──────────────────────────────────────────────
            elif msg["type"] == "audio_chunk" and listening:
                # Decode incoming base64 PCM float32 chunk
                chunk_bytes = base64.b64decode(msg["data"])
                chunk = np.frombuffer(chunk_bytes, dtype=np.float32)

                # Resample if client sends different SR
                client_sr = msg.get("sample_rate", CHUNK_SR)
                if client_sr != CHUNK_SR:
                    chunk = librosa.resample(
                        chunk, orig_sr=client_sr, target_sr=CHUNK_SR
                    )

                audio_buffer = np.append(audio_buffer, chunk)

                # Send live RMS + VAD
                rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
                voice = bool(is_voice_present(chunk, CHUNK_SR))
                await ws.send_text(json.dumps({
                    "type": "level",
                    "rms": round(rms, 4),
                    "voice": voice,
                }))

                # Process sliding windows
                while len(audio_buffer) >= window_samples:
                    window = audio_buffer[:window_samples]
                    audio_buffer = audio_buffer[step_samples:]

                    # Skip silent windows
                    if not is_voice_present(window, CHUNK_SR):
                        continue

                    # ── Text keyword gate ─────────────────────────────────
                    if keyword and not text_gate_open:
                        await ws.send_text(json.dumps({
                            "type": "gate_blocked",
                            "reason": "keyword_not_detected",
                            "keyword": keyword,
                        }))
                        continue

                    # ── Speaker verification (new DL pipeline) ────────────
                    live_embedding = get_embedding(window, CHUNK_SR)
                    if live_embedding is None:
                        continue  # skip if embedding extraction fails

                    # Cosine similarity against centroid (always computed)
                    similarity_score = cosine_similarity(centroid, live_embedding)

                    # Classifier prediction (if fine-tuned model exists)
                    classifier_result = fine_tuner.predict(speaker_id, live_embedding)

                    # Determine match based on available method
                    if classifier_result is not None:
                        # Fine-tuned classifier available → use it
                        threshold = speaker_profile.get(
                            "classifier_threshold",
                            DEFAULT_CLASSIFIER_THRESHOLD,
                        )
                        confidence = classifier_result["confidence"]
                        matched = confidence >= threshold
                        primary_score = confidence
                        matching_mode = "classifier"
                    else:
                        # Centroid similarity only
                        threshold = speaker_profile.get(
                            "similarity_threshold",
                            DEFAULT_SIMILARITY_THRESHOLD,
                        )
                        matched = similarity_score >= threshold
                        primary_score = similarity_score
                        confidence = 0.0
                        matching_mode = "similarity"

                    logger.debug(
                        f"Window — sim: {similarity_score:.3f}, "
                        f"clf: {confidence:.3f}, "
                        f"mode: {matching_mode}, match: {matched}"
                    )

                    await ws.send_text(json.dumps({
                        "type": "match",
                        "speaker": speaker_id,
                        "similarity_score": round(float(similarity_score), 4),
                        "classifier_score": round(float(confidence), 4) if classifier_result else None,
                        "fusion_score": round(float(primary_score), 4),
                        "matched": bool(matched),
                        "matching_mode": matching_mode,
                        "timestamp": datetime.now().isoformat(),
                        # Legacy compat fields
                        "dl_score": round(float(similarity_score), 4),
                        "dtw_dist": 0.0,
                        "dtw_score": 0.0,
                    }))

                    if matched and not alarm.is_active:
                        alarm.trigger(speaker_id, primary_score, similarity_score)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        alarm.remove_listener(send_alarm)
        if ws in active_connections:
            active_connections.remove(ws)


# ══════════════════════════════════════════════════════════════════════════════
# SERVE FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

import os

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
