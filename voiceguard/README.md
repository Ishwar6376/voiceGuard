# VoiceGuard — Voice Lock

A door lock that opens only for enrolled people, recognised by **who is speaking**. To stop someone from playing back a recording of your voice, you must also say a random phrase shown on screen.

## Run it

```bash
cd voiceguard/backend
python -m venv .venv && .venv\Scripts\activate      # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python main.py
```

Open http://localhost:8000 in **Chrome or Edge**. Their built-in speech recognition hears the challenge phrase, and it needs internet. The first start downloads the speaker model (~80 MB). The server only accepts connections from the same laptop.

## How to use

1. **Enroll**: type a name, press Record and read the sentence on screen (5 s). Keep talking the whole time; one word is not enough.
2. **Voice Notes**: record at least 2 more notes with the microphone and room you'll use at the door, then press **Calibrate**.
3. **Door**: press *Start Listening* and read the phrase shown (e.g. "basket window guitar") in your normal voice. The door unlocks for 5 s, then locks again. Every attempt appears in the Access Log.

Repeat 1–2 for every person who should be allowed in. In **Settings** you can turn the challenge phrase off (e.g. for a demo without internet), change the unlock time, or adjust a person's threshold.

## How it works

```
mic → speech check → 1.5 s windows → ECAPA-TDNN → voiceprint (192 numbers)
    → compare with every enrolled person → same person in 2 of last 3 windows
    + current phrase heard → 🔓 unlock (auto re-lock) → access log
```

| Step | What happens |
|---|---|
| Speech check | A window is scored only if it has ≥ 0.5 s of voiced sound, is not noise-like (hiss, fan) and has most energy between 80 and 4000 Hz (rejects rumble and hum). |
| Voiceprint | A pretrained deep learning model ([ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), trained on 7,000+ speakers) turns 1.5 s of speech into 192 numbers describing the voice: vocal-tract resonances, timbre and speaking style, not just pitch. It is used as-is; nothing is retrained. |
| Identification | Cosine similarity to each enrolled person's average voiceprint. The same person typically scores 0.5–0.8, different people 0–0.3. The best match must pass that person's threshold. |
| Decision | Unlock when the **same** person wins 2 of the last 3 speech windows **and** the current random phrase was heard. If the phrase is heard but the voice doesn't match within 3 s: **denied**, and a new phrase is shown. |
| Anti-replay | A fresh phrase from 48 words (103,776 combinations) after every attempt, so an old recording never contains the right words. |
| Calibration | Each voice note is scored against a voiceprint built from the *other* notes. The threshold is set so ~90% of those held-out windows pass (never below 0.45), and above the scores of other enrolled people. |

## Measured results

Held-out recordings of real speakers (LibriSpeech), not seen during enrollment:

| | Similarity |
|---|---|
| Same speaker | 0.47 – 0.73 |
| Different speaker | 0.03 – 0.19 |

End-to-end in Chrome: the enrolled speaker unlocked the door, an unenrolled speaker never did, noise was ignored, and old phrases were rejected.

## Limitations

- Needs Chrome/Edge with internet for the challenge phrase; otherwise turn it off in Settings.
- Very similar voices (e.g. twins) may be confused. Calibrating with both enrolled helps.
- The lock is virtual (on screen). To drive a real relay or solenoid, act on the state change in `backend/lock_manager.py`.

## Data

Recordings, profiles, the config and the access log live in `backend/storage/` and are git-ignored; recordings are biometric data.
