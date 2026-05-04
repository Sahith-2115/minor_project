import numpy as np
import librosa

# ---------------- CONFIG ---------------- #
TARGET_SR = 16000
DURATION = 4  # seconds
TARGET_LENGTH = TARGET_SR * DURATION  # 64000 samples


# ---------------- LOAD AUDIO ---------------- #
def load_audio(file_path):
    """
    Load audio file as mono
    """
    signal, sr = librosa.load(file_path, sr=None, mono=True)
    return signal, sr


# ---------------- RESAMPLE ---------------- #
def resample_audio(signal, orig_sr, target_sr=TARGET_SR):
    if orig_sr != target_sr:
        signal = librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
    return signal


# ---------------- FIX LENGTH ---------------- #
def fix_length(signal, target_length=TARGET_LENGTH):
    """
    Pad or trim signal to fixed length
    """
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        pad_length = target_length - len(signal)
        signal = np.pad(signal, (0, pad_length), mode='constant')

    return signal


# ---------------- NORMALIZE ---------------- #
def normalize_audio(signal):
    """
    Normalize to [-1, 1]
    """
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal


# ---------------- FULL PIPELINE ---------------- #
def preprocess_audio(file_path):
    signal, sr = load_audio(file_path)

    signal = resample_audio(signal, sr)
    signal = fix_length(signal)
    signal = normalize_audio(signal)

    return signal


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    from data_loader import load_icbhi_dataset

    df = load_icbhi_dataset()

    print("\nTesting preprocessing on first 3 samples...\n")

    for i in range(3):
        path = df.iloc[i]["file_path"]
        processed_signal = preprocess_audio(path)

        print(f"Sample {i+1}")
        print(f"Length: {len(processed_signal)}")
        print(f"Min: {np.min(processed_signal):.4f}, Max: {np.max(processed_signal):.4f}")
        print("-" * 40)