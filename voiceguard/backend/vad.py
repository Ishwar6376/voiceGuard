"""
vad.py — Voice Activity Detection

A window counts as speech only if it
  1. contains enough voiced sound (≥ MIN_VOICED_S of loud 25 ms frames) —
     a click, a cough or the tail of a word is not enough to judge a voice,
  2. is not noise-like (spectral flatness of the loud frames: hiss/fans are
     flat, voices are peaky),
  3. has most energy in the voice range (80–4000 Hz): rejects rumble,
     traffic and mains hum. (Not 300–3400 Hz "telephone band" — deep male
     voices keep most of their energy below 300 Hz and would be rejected.)

This matters because the speaker encoder produces a (fairly similar)
embedding for ANY sound — mostly-silent windows end up describing the room
and microphone rather than the person, so they must not be scored.
"""
import numpy as np
import librosa

RMS_THRESHOLD         = 0.006   # low enough for phone speaker → room → mic
VOICED_FRAME_RMS      = 0.01    # a 25 ms frame this loud counts as voiced (-40 dBFS)
MIN_VOICED_S          = 0.5     # voiced sound needed in a 1.5 s window
MAX_FLATNESS          = 0.25
VOICE_BAND_HZ         = (80, 4000)
MIN_VOICE_BAND_RATIO  = 0.70

_N_FFT = 512
_HOP   = 160   # 10 ms


def is_voice_present(audio: np.ndarray, sr: int, threshold: float = RMS_THRESHOLD) -> bool:
    """Energy-only check: is there any sound at all?"""
    if len(audio) == 0:
        return False
    return float(np.sqrt(np.mean(audio ** 2))) > threshold


def voiced_seconds(audio: np.ndarray, sr: int) -> float:
    """Total duration of loud (voiced) 25 ms frames."""
    frame_rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=_HOP, center=False)[0]
    return float((frame_rms > VOICED_FRAME_RMS).sum()) * _HOP / sr


def is_speech(audio: np.ndarray, sr: int, min_voiced_s: float = MIN_VOICED_S) -> bool:
    """
    Energy + duration + spectral checks: does this sound like a human voice?
    Use min_voiced_s=0 for short live chunks (level indicator only).
    """
    if not is_voice_present(audio, sr):
        return False
    if len(audio) < _N_FFT:
        audio = np.pad(audio, (0, _N_FFT - len(audio)))

    if min_voiced_s > 0 and voiced_seconds(audio, sr) < min_voiced_s:
        return False

    power = np.abs(librosa.stft(audio, n_fft=_N_FFT, hop_length=_HOP)) ** 2
    total = power.sum()
    if total <= 0:
        return False

    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    lo, hi = VOICE_BAND_HZ
    band_ratio = power[(freqs >= lo) & (freqs <= hi)].sum() / total

    # Weight each frame's flatness by its energy: the quiet gaps between
    # syllables are dominated by background hiss and would otherwise make
    # real speech in a slightly noisy room look like noise.
    frame_flatness = librosa.feature.spectral_flatness(S=np.sqrt(power))[0]
    flatness = float(np.average(frame_flatness, weights=power.sum(axis=0)))

    return band_ratio >= MIN_VOICE_BAND_RATIO and flatness <= MAX_FLATNESS
