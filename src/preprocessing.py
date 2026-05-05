import numpy as np
import librosa

TARGET_SR = 16000
DURATION = 2  # seconds
TARGET_LENGTH = TARGET_SR * DURATION


def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=None, mono=True)
    return signal, sr


def resample_audio(signal, orig_sr, target_sr=TARGET_SR):
    if orig_sr != target_sr:
        signal = librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
    return signal


def fix_length(signal, target_length=TARGET_LENGTH):
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        pad_length = target_length - len(signal)
        signal = np.pad(signal, (0, pad_length), mode='constant')
    return signal


def normalize_audio(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal


def preprocess_audio(file_path):
    signal, sr = load_audio(file_path)
    signal = resample_audio(signal, sr)
    signal = fix_length(signal)
    signal = normalize_audio(signal)
    return signal


def preprocess_audio_from_signal(signal):
    signal = fix_length(signal)
    signal = normalize_audio(signal)
    return signal