"""
vad.py — Voice Activity Detection
Trims silence from audio using energy + zero-crossing rate.
"""
import numpy as np
import librosa


def trim_silence(audio: np.ndarray, sr: int, energy_threshold: float = 0.01, frame_length: int = 512, hop_length: int = 128) -> np.ndarray:
    """
    Remove leading/trailing silence using RMS energy.
    Returns trimmed audio array.
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    above = np.where(rms > energy_threshold)[0]

    if len(above) == 0:
        return audio  # all silence, return as-is

    start_frame = max(0, above[0] - 2)
    end_frame = min(len(rms) - 1, above[-1] + 2)

    start_sample = start_frame * hop_length
    end_sample = min(len(audio), end_frame * hop_length + frame_length)
    return audio[start_sample:end_sample]


def is_voice_present(audio: np.ndarray, sr: int, threshold: float = 0.006) -> bool:
    """
    Quick VAD check — returns True if audio contains voice energy.
    Threshold lowered to 0.006 to handle phone speaker → room → mic scenarios.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    return rms > threshold

