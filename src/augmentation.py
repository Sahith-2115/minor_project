import numpy as np
import librosa
import random


def add_noise(signal, noise_factor=0.003):
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise


def time_stretch(signal):
    # VERY small variation only
    return librosa.effects.time_stretch(signal, rate=random.uniform(0.95, 1.05))


def pitch_shift(signal, sr):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=random.uniform(-1, 1))


def augment_audio(signal, sr=16000):
    """
    Apply ONLY ONE augmentation at a time (important)
    """

    choice = random.choice(["noise", "stretch", "pitch", "none"])

    if choice == "noise":
        signal = add_noise(signal)
    elif choice == "stretch":
        signal = time_stretch(signal)
    elif choice == "pitch":
        signal = pitch_shift(signal, sr)

    return signal