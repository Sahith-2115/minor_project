import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
DATA_PATH = "/home/sahith/projects/minor_project_408/data/raw/audio_and_text_files"
TARGET_SR = 16000

# Label mapping
LABEL_MAP = {
    (0, 0): 0,  # Normal
    (1, 0): 1,  # Crackles
    (0, 1): 2,  # Wheezes
    (1, 1): 3   # Both
}

CLASS_NAMES = {
    0: "Normal",
    1: "Crackles",
    2: "Wheezes",
    3: "Both"
}


# ---------------- LOAD AUDIO ---------------- #
def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=None, mono=True)
    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
    return signal


# ---------------- EXTRACT CYCLES ---------------- #
def extract_cycles(wav_path, txt_path):
    signal = load_audio(wav_path)

    cycles = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 4:
            continue

        start = float(parts[0])
        end = float(parts[1])
        crackles = int(parts[2])
        wheezes = int(parts[3])

        start_sample = int(start * TARGET_SR)
        end_sample = int(end * TARGET_SR)

        segment = signal[start_sample:end_sample]

        if len(segment) < 100:  # skip very short segments
            continue

        label = LABEL_MAP[(crackles, wheezes)]

        cycles.append((segment, label))

    return cycles


# ---------------- MAIN LOADER ---------------- #
def load_icbhi_cycles(data_path=DATA_PATH):
    data = []

    files = os.listdir(data_path)
    wav_files = [f for f in files if f.endswith(".wav")]

    print(f"Total audio files: {len(wav_files)}")

    for wav_file in tqdm(wav_files):
        wav_path = os.path.join(data_path, wav_file)
        txt_path = wav_path.replace(".wav", ".txt")

        if not os.path.exists(txt_path):
            continue

        patient_id = wav_file.split("_")[0]

        cycles = extract_cycles(wav_path, txt_path)

        for segment, label in cycles:
            data.append({
                "signal": segment,
                "label": label,
                "patient_id": patient_id,
                "class_name": CLASS_NAMES[label]
            })

    df = pd.DataFrame(data)

    print("\nCycle-level dataset summary:")
    print(df["class_name"].value_counts())

    return df


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    df = load_icbhi_cycles()

    print("\nSample:")
    print(df.head())

    print("\nTotal cycles:", len(df))