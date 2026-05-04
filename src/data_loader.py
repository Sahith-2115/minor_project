import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
DATA_PATH = "/home/sahith/projects/minor_project_408/data/raw/audio_and_text_files"

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


# ---------------- HELPER FUNCTION ---------------- #
def parse_annotation(txt_file):
    """
    Reads annotation file and determines presence of crackles and wheezes.
    """
    crackles = 0
    wheezes = 0

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 4:
            continue

        c = int(parts[2])
        w = int(parts[3])

        if c == 1:
            crackles = 1
        if w == 1:
            wheezes = 1

    return crackles, wheezes


# ---------------- MAIN LOADER ---------------- #
def load_icbhi_dataset(data_path=DATA_PATH):
    data = []

    files = os.listdir(data_path)

    wav_files = [f for f in files if f.endswith(".wav")]

    print(f"Total audio files found: {len(wav_files)}")

    for wav_file in tqdm(wav_files):
        wav_path = os.path.join(data_path, wav_file)

        txt_file = wav_file.replace(".wav", ".txt")
        txt_path = os.path.join(data_path, txt_file)

        if not os.path.exists(txt_path):
            continue

        crackles, wheezes = parse_annotation(txt_path)

        label = LABEL_MAP[(crackles, wheezes)]

        data.append({
            "file_path": wav_path,
            "crackles": crackles,
            "wheezes": wheezes,
            "label": label,
            "class_name": CLASS_NAMES[label]
        })

    df = pd.DataFrame(data)

    print("\nDataset Summary:")
    print(df["class_name"].value_counts())

    return df


# ---------------- RUN TEST ---------------- #
if __name__ == "__main__":
    df = load_icbhi_dataset()
    print("\nSample data:")
    print(df.head())