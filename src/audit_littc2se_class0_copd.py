import os
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

METADATA_FILES = {
    "train": os.path.join(
        PROJECT_ROOT,
        "data/processed/unified_dataset/train_metadata.csv"
    ),
    "validation": os.path.join(
        PROJECT_ROOT,
        "data/processed/unified_dataset/validation_metadata.csv"
    ),
    "test": os.path.join(
        PROJECT_ROOT,
        "data/processed/unified_dataset/test_metadata.csv"
    )
}

PREDICTION_FILES = {
    "validation": os.path.join(
        PROJECT_ROOT,
        "outputs/ast_exp4/reports/patient_134_audit/device_audit/validation_predictions_exp4.csv"
    ),
    "test": os.path.join(
        PROJECT_ROOT,
        "outputs/ast_exp4/reports/test_predictions.csv"
    )
}

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/littc2se_class0_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_device(path):
    if pd.isna(path):
        return "UNKNOWN"

    filename = os.path.basename(str(path))

    if filename.lower().endswith(".wav"):
        filename = filename[:-4]

    parts = filename.split("_")

    if len(parts) >= 5:
        return parts[-1]

    return "UNKNOWN"

def load_split(split):
    metadata = pd.read_csv(
        METADATA_FILES[split]
    )

    if split == "train":
        metadata["correct"] = np.nan
        metadata["predicted_label"] = np.nan
        return metadata

    predictions = pd.read_csv(
        PREDICTION_FILES[split]
    )

    if "predicted_label" in predictions.columns:
        prediction_column = "predicted_label"
    elif "predicted_class" in predictions.columns:
        prediction_column = "predicted_class"
    else:
        print(
            f"ERROR: No prediction column found in {split}"
        )
        print(
            f"Available columns: {list(predictions.columns)}"
        )
        sys.exit(1)

    predictions = predictions[
        [
            "sample_id",
            prediction_column
        ]
    ].copy()

    predictions = predictions.rename(
        columns={
            prediction_column: "predicted_label"
        }
    )

    predictions = predictions.drop_duplicates(
        subset=["sample_id"]
    )

    merged = metadata.merge(
        predictions,
        on="sample_id",
        how="left"
    )

    merged["correct"] = (
        merged["class_name"] ==
        merged["predicted_label"]
    )

    return merged

def get_audio_features(path):
    try:
        audio, sr = librosa.load(
            path,
            sr=None,
            mono=True
        )

        if len(audio) == 0:
            return None

        duration = len(audio) / sr

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        zcr = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    audio
                )
            )
        )

        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=audio,
                    sr=sr
                )
            )
        )

        bandwidth = float(
            np.mean(
                librosa.feature.spectral_bandwidth(
                    y=audio,
                    sr=sr
                )
            )
        )

        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=audio,
                    sr=sr,
                    roll_percent=0.85
                )
            )
        )

        return {
            "sample_rate": sr,
            "audio_duration": duration,
            "rms": rms,
            "peak": peak,
            "zcr": zcr,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff
        }

    except Exception as e:
        print(
            f"ERROR loading {path}: {e}"
        )
        return None

def create_cycle_dataset():
    validation = load_split("validation")
    test = load_split("test")

    combined = pd.concat(
        [
            validation,
            test
        ],
        ignore_index=True
    )

    data = combined[
        (combined["source"] == "ICBHI") &
        (combined["class_name"] == "COPD") &
        (combined["sound_class"] == 0)
    ].copy()

    data["device"] = data[
        "audio_path"
    ].apply(
        extract_device
    )

    print()
    print("LittC2SE COPD CLASS 0 AUDIT")
    print(
        f"Total LittC2SE class 0 cycles: "
        f"{len(data[data['device'] == 'LittC2SE'])}"
    )

    data = data[
        data["device"] == "LittC2SE"
    ].copy()

    print(
        f"Unique patient groups: "
        f"{data['patient_group'].nunique()}"
    )

    print(
        f"Unique recordings: "
        f"{data['audio_path'].nunique()}"
    )

    return data

def create_patient_summary(data):
    rows = []

    for patient, group in data.groupby(
        "patient_group"
    ):
        rows.append({
            "patient_group": patient,
            "split": group["split"].iloc[0],
            "cycles": len(group),
            "audio_files": group["audio_path"].nunique(),
            "correct": int(
                group["correct"].sum()
            ),
            "errors": int(
                (~group["correct"]).sum()
            ),
            "accuracy": (
                group["correct"].mean()
                if group["correct"].notna().any()
                else np.nan
            ),
            "mean_cycle_duration": (
                group["duration"].mean()
            ),
            "median_cycle_duration": (
                group["duration"].median()
            ),
            "mean_start_time": (
                group["start_time"].mean()
            ),
            "mean_end_time": (
                group["end_time"].mean()
            )
        })

    summary = pd.DataFrame(rows)

    return summary

def create_audio_dataset(data):
    unique_audio = (
        data[
            [
                "audio_path",
                "patient_group",
                "split",
                "correct",
                "predicted_label"
            ]
        ]
        .drop_duplicates(
            subset=["audio_path"]
        )
        .copy()
    )

    rows = []

    print()
    print(
        f"Extracting acoustic features from "
        f"{len(unique_audio)} recordings..."
    )

    for _, row in tqdm(
        unique_audio.iterrows(),
        total=len(unique_audio),
        desc="Audio analysis"
    ):
        features = get_audio_features(
            row["audio_path"]
        )

        if features is None:
            continue

        result = {
            "audio_path": row["audio_path"],
            "patient_group": row["patient_group"],
            "split": row["split"],
            "correct": row["correct"],
            "predicted_label": row["predicted_label"]
        }

        result.update(features)

        rows.append(result)

    return pd.DataFrame(rows)

def create_patient_audio_summary(
    audio_features
):
    if len(audio_features) == 0:
        return pd.DataFrame()

    numeric_columns = [
        "audio_duration",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "sample_rate"
    ]

    rows = []

    for patient, group in audio_features.groupby(
        "patient_group"
    ):
        row = {
            "patient_group": patient,
            "split": group["split"].iloc[0],
            "audio_files": len(group)
        }

        if group["correct"].notna().any():
            row["cycle_level_accuracy"] = (
                group["correct"].mean()
            )

        for column in numeric_columns:
            row[f"{column}_mean"] = (
                group[column].mean()
            )
            row[f"{column}_median"] = (
                group[column].median()
            )
            row[f"{column}_std"] = (
                group[column].std()
            )
            row[f"{column}_min"] = (
                group[column].min()
            )
            row[f"{column}_max"] = (
                group[column].max()
            )

        rows.append(row)

    return pd.DataFrame(rows)

def compare_correct_vs_error(
    audio_features
):
    print()
    print(
        "CORRECT VS ERROR AUDIO CHARACTERISTICS"
    )

    if len(audio_features) == 0:
        return pd.DataFrame()

    test_validation = audio_features[
        audio_features["correct"].notna()
    ].copy()

    rows = []

    for column in [
        "audio_duration",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff"
    ]:
        correct_values = test_validation[
            test_validation["correct"] == True
        ][column]

        error_values = test_validation[
            test_validation["correct"] == False
        ][column]

        rows.append({
            "feature": column,
            "correct_count": len(
                correct_values
            ),
            "correct_mean": (
                correct_values.mean()
            ),
            "correct_median": (
                correct_values.median()
            ),
            "error_count": len(
                error_values
            ),
            "error_mean": (
                error_values.mean()
            ),
            "error_median": (
                error_values.median()
            )
        })

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return summary

def create_patient_classification_summary(
    patient_summary
):
    print()
    print(
        "PATIENT LEVEL RESULTS"
    )

    display_columns = [
        "patient_group",
        "split",
        "cycles",
        "audio_files",
        "correct",
        "errors",
        "accuracy"
    ]

    available = [
        column
        for column in display_columns
        if column in patient_summary.columns
    ]

    result = patient_summary[
        available
    ].sort_values(
        [
            "accuracy",
            "cycles"
        ],
        na_position="last"
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return result

def main():
    data = create_cycle_dataset()

    if len(data) == 0:
        print(
            "ERROR: No LittC2SE COPD class 0 samples found."
        )
        sys.exit(1)

    patient_summary = create_patient_summary(
        data
    )

    print()
    print(
        "PATIENT DISTRIBUTION"
    )

    print(
        patient_summary[
            [
                "patient_group",
                "split",
                "cycles",
                "audio_files",
                "correct",
                "errors",
                "accuracy"
            ]
        ].sort_values(
            [
                "accuracy",
                "cycles"
            ],
            na_position="last"
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    audio_features = create_audio_dataset(
        data
    )

    audio_summary = create_patient_audio_summary(
        audio_features
    )

    classification_summary = (
        create_patient_classification_summary(
            patient_summary
        )
    )

    correct_error_summary = (
        compare_correct_vs_error(
            audio_features
        )
    )

    data.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_cycles.csv"
        ),
        index=False
    )

    patient_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_patient_summary.csv"
        ),
        index=False
    )

    audio_features.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_audio_features.csv"
        ),
        index=False
    )

    audio_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_patient_audio_summary.csv"
        ),
        index=False
    )

    classification_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_classification_summary.csv"
        ),
        index=False
    )

    correct_error_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_correct_vs_error.csv"
        ),
        index=False
    )

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print()
    print("Generated files:")

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):
        print(filename)

if __name__ == "__main__":
    main()