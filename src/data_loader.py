import os
import pandas as pd
import librosa
from tqdm import tqdm


# ---------------- PATHS ---------------- #
DATA_PATH = "/home/sahith/projects/minor_project_408/data/raw/audio_and_text_files"

DIAGNOSIS_PATH = "/home/sahith/projects/minor_project_408/data/raw/patient_diagnosis.csv"


TARGET_SR = 16000


# ---------------- DISEASE LABELS ---------------- #
DISEASE_MAP = {
    "Healthy": 0,
    "COPD": 1,
    "Pneumonia": 2
}


# ---------------- LOAD DIAGNOSIS ---------------- #
def load_diagnosis():

    df = pd.read_csv(DIAGNOSIS_PATH)

    diagnosis_dict = {}

    for _, row in df.iterrows():

        patient_id = str(row["ID"]).strip()

        disease = str(row["Symptom"]).strip()

        if disease in DISEASE_MAP:

            diagnosis_dict[patient_id] = disease

    return diagnosis_dict


# ---------------- LOAD AUDIO ---------------- #
def load_audio(file_path):

    signal, sr = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    if sr != TARGET_SR:

        signal = librosa.resample(
            signal,
            orig_sr=sr,
            target_sr=TARGET_SR
        )

    return signal


# ---------------- EXTRACT RESPIRATORY CYCLES ---------------- #
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

        start_sample = int(start * TARGET_SR)

        end_sample = int(end * TARGET_SR)

        segment = signal[start_sample:end_sample]

        # skip tiny segments
        if len(segment) < 100:
            continue

        cycles.append(segment)

    return cycles


# ---------------- MAIN LOADER ---------------- #
def load_icbhi_cycles():

    diagnosis_dict = load_diagnosis()

    data = []

    files = os.listdir(DATA_PATH)

    wav_files = [
        f for f in files
        if f.endswith(".wav")
    ]

    print(f"\nTotal audio files found: {len(wav_files)}")

    for wav_file in tqdm(wav_files):

        patient_id = wav_file.split("_")[0]

        # Skip diseases not in selected classes
        if patient_id not in diagnosis_dict:
            continue

        disease = diagnosis_dict[patient_id]

        label = DISEASE_MAP[disease]

        wav_path = os.path.join(
            DATA_PATH,
            wav_file
        )

        txt_path = wav_path.replace(
            ".wav",
            ".txt"
        )

        if not os.path.exists(txt_path):
            continue

        cycles = extract_cycles(
            wav_path,
            txt_path
        )

        for segment in cycles:

            data.append({

                "signal": segment,

                "label": label,

                "disease": disease,

                "patient_id": patient_id
            })

    df = pd.DataFrame(data)

    print("\nDisease Distribution:")
    print(df["disease"].value_counts())
    print("\nTotal Samples:", len(df))

    return df


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    df = load_icbhi_cycles()
    print("\nSample Data:")
    print(df.head())