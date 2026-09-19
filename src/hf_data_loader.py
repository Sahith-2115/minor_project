import os
import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm


# ---------------- PATHS ---------------- #
TRAIN_PATH = "/home/sahith/projects/minor_project_408/data/raw/HF_Lung_V1-master/train"
TEST_PATH = "/home/sahith/projects/minor_project_408/data/raw/HF_Lung_V1-master/test"

TARGET_SR = 16000
WINDOW_DURATION = 2.0

WINDOW_SIZE = int(
    TARGET_SR * WINDOW_DURATION
)

# ---------------- LABEL MAP ---------------- #
LABEL_MAP = {
    "Normal": 0,
    "Crackles": 1,
    "Wheezes": 2,
    "Both": 3
}


# ---------------- TIME CONVERSION ---------------- #
def time_to_seconds(t):

    parts = t.split(":")

    h = int(parts[0])

    m = int(parts[1])

    s = float(parts[2])

    return h * 3600 + m * 60 + s


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

    signal = signal / (
        np.max(np.abs(signal)) + 1e-6
    )

    return signal


# ---------------- PARSE LABEL FILE ---------------- #
def parse_label_file(label_path):

    crackles = []

    wheezes = []

    with open(label_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 3:
            continue

        event_type = parts[0]

        start = time_to_seconds(parts[1])

        end = time_to_seconds(parts[2])

        # D = discontinuous = crackles
        if event_type == "D":

            crackles.append((start, end))

        # C = continuous = wheezes
        elif event_type == "C":

            wheezes.append((start, end))

    return crackles, wheezes


# ---------------- OVERLAP CHECK ---------------- #
def has_overlap(start1, end1, start2, end2):

    return max(start1, start2) < min(end1, end2)


# ---------------- ASSIGN LABEL ---------------- #
def assign_label(
    seg_start,
    seg_end,
    crackles,
    wheezes
):

    has_crackle = False

    has_wheeze = False

    for s, e in crackles:

        if has_overlap(
            seg_start,
            seg_end,
            s,
            e
        ):

            has_crackle = True

            break

    for s, e in wheezes:

        if has_overlap(
            seg_start,
            seg_end,
            s,
            e
        ):

            has_wheeze = True

            break

    if has_crackle and has_wheeze:

        return "Both"

    elif has_crackle:

        return "Crackles"

    elif has_wheeze:

        return "Wheezes"

    else:

        return "Normal"


# ---------------- PROCESS FOLDER ---------------- #
def process_folder(folder_path):

    data = []

    files = os.listdir(folder_path)

    wav_files = [
        f for f in files
        if f.endswith(".wav")
    ]

    print(f"\nProcessing {folder_path}")

    print(f"Total wav files: {len(wav_files)}")

    for wav_file in tqdm(wav_files):

        wav_path = os.path.join(
            folder_path,
            wav_file
        )

        label_path = os.path.join(
            folder_path,
            wav_file.replace(
                ".wav",
                "_label.txt"
            )
        )

        if not os.path.exists(label_path):
            continue

        signal = load_audio(wav_path)

        crackles, wheezes = parse_label_file(
            label_path
        )

        duration = len(signal) / TARGET_SR

        patient_id = wav_file.split("_")[1]

        start = 0.0

        while start + WINDOW_DURATION <= duration:

            end = start + WINDOW_DURATION

            start_sample = int(start * TARGET_SR)

            end_sample = int(end * TARGET_SR)

            segment = signal[
                start_sample:end_sample
            ]

            if len(segment) != WINDOW_SIZE:

                start += WINDOW_DURATION

                continue

            class_name = assign_label(
                start,
                end,
                crackles,
                wheezes
            )

            label = LABEL_MAP[class_name]

            data.append({

                "signal": segment,

                "label": label,

                "class_name": class_name,

                "patient_id": patient_id
            })

            start += WINDOW_DURATION

    return data


# ---------------- MAIN LOADER ---------------- #
def load_hf_dataset():

    train_data = process_folder(TRAIN_PATH)

    test_data = process_folder(TEST_PATH)

    all_data = train_data + test_data

    df = pd.DataFrame(all_data)

    print("\nDataset Summary:")

    print(df["class_name"].value_counts())

    print("\nTotal Samples:")

    print(len(df))

    return df


# ---------------- TEST ---------------- #
if __name__ == "__main__":

    df = load_hf_dataset()

    print("\nSample Data:")

    print(df.head())