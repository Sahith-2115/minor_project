import os
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

TEST_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/test_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/visualizations"
)

PATIENT_134 = "ICBHI_134"
REFERENCE_PATIENT = "ICBHI_163"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

def normalize_audio(y):
    max_value = np.max(np.abs(y))
    if max_value == 0:
        return y
    return y / max_value

def save_waveform(y, sr, title, output_path):
    plt.figure(figsize=(14, 4))
    time = np.arange(len(y)) / sr
    plt.plot(time, y)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def save_melspectrogram(y, sr, title, output_path):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmax=min(8000, sr // 2)
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=min(8000, sr // 2)
    )
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def save_comparison(
    y1,
    sr1,
    y2,
    sr2,
    title1,
    title2,
    output_path
):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    time1 = np.arange(len(y1)) / sr1
    time2 = np.arange(len(y2)) / sr2

    axes[0].plot(time1, y1)
    axes[0].set_title(title1)
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time2, y2)
    axes[1].set_title(title2)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def save_mel_comparison(
    y1,
    sr1,
    y2,
    sr2,
    title1,
    title2,
    output_path
):
    mel1 = librosa.feature.melspectrogram(
        y=y1,
        sr=sr1,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmax=min(8000, sr1 // 2)
    )

    mel2 = librosa.feature.melspectrogram(
        y=y2,
        sr=sr2,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmax=min(8000, sr2 // 2)
    )

    mel1_db = librosa.power_to_db(
        mel1,
        ref=np.max
    )

    mel2_db = librosa.power_to_db(
        mel2,
        ref=np.max
    )

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    img1 = librosa.display.specshow(
        mel1_db,
        sr=sr1,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=min(8000, sr1 // 2),
        ax=axes[0]
    )

    axes[0].set_title(title1)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    img2 = librosa.display.specshow(
        mel2_db,
        sr=sr2,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=min(8000, sr2 // 2),
        ax=axes[1]
    )

    axes[1].set_title(title2)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def audio_summary(y, sr):
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                y,
                frame_length=min(2048, len(y)),
                hop_length=min(512, max(1, len(y) // 4))
            )
        )
    )

    centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )
    )

    bandwidth = float(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=y,
                sr=sr
            )
        )
    )

    rolloff = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=y,
                sr=sr
            )
        )
    )

    return {
        "sampling_rate": sr,
        "duration": len(y) / sr,
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff
    }

def main():
    print(f"Loading metadata from: {TEST_METADATA}")

    if not os.path.exists(TEST_METADATA):
        print("ERROR: Test metadata file not found.")
        sys.exit(1)

    metadata = pd.read_csv(TEST_METADATA)

    required_columns = [
        "audio_path",
        "source",
        "class_name",
        "patient_group"
    ]

    missing = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    icbhi_copd = metadata[
        (metadata["source"] == "ICBHI") &
        (metadata["class_name"] == "COPD")
    ].copy()

    patient_134 = icbhi_copd[
        icbhi_copd["patient_group"] == PATIENT_134
    ].copy()

    reference = icbhi_copd[
        icbhi_copd["patient_group"] == REFERENCE_PATIENT
    ].copy()

    print()
    print(f"{PATIENT_134} cycles: {len(patient_134)}")
    print(f"{PATIENT_134} audio files: {patient_134['audio_path'].nunique()}")
    print()
    print(f"{REFERENCE_PATIENT} cycles: {len(reference)}")
    print(f"{REFERENCE_PATIENT} audio files: {reference['audio_path'].nunique()}")

    patient_134_paths = sorted(
        patient_134["audio_path"].dropna().unique()
    )

    reference_paths = sorted(
        reference["audio_path"].dropna().unique()
    )

    print()
    print("Patient 134 recordings:")
    for path in patient_134_paths:
        print(path)

    print()
    print(f"{REFERENCE_PATIENT} recordings:")
    for path in reference_paths:
        print(path)

    all_rows = []

    print()
    print("Analyzing Patient 134 recordings")

    for index, path in enumerate(
        tqdm(
            patient_134_paths,
            desc="Patient 134",
            unit="file"
        ),
        start=1
    ):
        y, sr = load_audio(path)
        features = audio_summary(y, sr)

        sample = patient_134[
            patient_134["audio_path"] == path
        ]

        all_rows.append({
            "patient_group": PATIENT_134,
            "audio_path": path,
            "cycle_count": len(sample),
            **features
        })

        base = f"patient_134_{index}"

        save_waveform(
            y,
            sr,
            f"Patient 134 Recording {index} Waveform",
            os.path.join(
                OUTPUT_DIR,
                f"{base}_waveform.png"
            )
        )

        save_melspectrogram(
            y,
            sr,
            f"Patient 134 Recording {index} Mel Spectrogram",
            os.path.join(
                OUTPUT_DIR,
                f"{base}_mel.png"
            )
        )

    print()
    print(f"Analyzing {REFERENCE_PATIENT} recordings")

    for index, path in enumerate(
        tqdm(
            reference_paths,
            desc=f"{REFERENCE_PATIENT}",
            unit="file"
        ),
        start=1
    ):
        y, sr = load_audio(path)
        features = audio_summary(y, sr)

        sample = reference[
            reference["audio_path"] == path
        ]

        all_rows.append({
            "patient_group": REFERENCE_PATIENT,
            "audio_path": path,
            "cycle_count": len(sample),
            **features
        })

        base = f"reference_{index}"

        save_waveform(
            y,
            sr,
            f"{REFERENCE_PATIENT} Recording {index} Waveform",
            os.path.join(
                OUTPUT_DIR,
                f"{base}_waveform.png"
            )
        )

        save_melspectrogram(
            y,
            sr,
            f"{REFERENCE_PATIENT} Recording {index} Mel Spectrogram",
            os.path.join(
                OUTPUT_DIR,
                f"{base}_mel.png"
            )
        )

    if patient_134_paths and reference_paths:
        y134, sr134 = load_audio(patient_134_paths[0])
        yref, srref = load_audio(reference_paths[0])

        save_comparison(
            y134,
            sr134,
            yref,
            srref,
            f"{PATIENT_134} Recording 1",
            f"{REFERENCE_PATIENT} Recording 1",
            os.path.join(
                OUTPUT_DIR,
                "patient_134_vs_reference_waveform.png"
            )
        )

        save_mel_comparison(
            y134,
            sr134,
            yref,
            srref,
            f"{PATIENT_134} Recording 1 Mel Spectrogram",
            f"{REFERENCE_PATIENT} Recording 1 Mel Spectrogram",
            os.path.join(
                OUTPUT_DIR,
                "patient_134_vs_reference_melspectrogram.png"
            )
        )

    results = pd.DataFrame(all_rows)

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "patient_134_vs_reference_audio_features.csv"
        ),
        index=False
    )

    print()
    print("Audio feature comparison")
    print()

    numeric_columns = [
        "duration",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff"
    ]

    summary = results.groupby(
        "patient_group"
    )[numeric_columns].agg(
        ["mean", "median"]
    )

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("Generated files:")

    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(filename)

if __name__ == "__main__":
    main()