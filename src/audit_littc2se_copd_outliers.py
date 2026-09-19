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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/device_audit/littc2se_analysis"
)

TARGET_PATIENTS = [
    "ICBHI_134",
    "ICBHI_199",
    "ICBHI_141"
]

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

def audio_features(path):
    try:
        y, sr = librosa.load(
            path,
            sr=None,
            mono=True
        )

        if len(y) == 0:
            return {
                "sampling_rate": sr,
                "duration": np.nan,
                "rms": np.nan,
                "peak": np.nan,
                "zcr": np.nan,
                "spectral_centroid": np.nan,
                "spectral_bandwidth": np.nan,
                "spectral_rolloff": np.nan,
                "status": "empty"
            }

        rms = float(
            np.sqrt(
                np.mean(y ** 2)
            )
        )

        peak = float(
            np.max(
                np.abs(y)
            )
        )

        frame_length = min(
            2048,
            len(y)
        )

        hop_length = min(
            512,
            max(
                1,
                len(y) // 4
            )
        )

        zcr = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    y,
                    frame_length=frame_length,
                    hop_length=hop_length
                )
            )
        )

        spectral_centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr
                )
            )
        )

        spectral_bandwidth = float(
            np.mean(
                librosa.feature.spectral_bandwidth(
                    y=y,
                    sr=sr
                )
            )
        )

        spectral_rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=y,
                    sr=sr,
                    roll_percent=0.85
                )
            )
        )

        return {
            "sampling_rate": sr,
            "duration": float(
                len(y) / sr
            ),
            "rms": rms,
            "peak": peak,
            "zcr": zcr,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "status": "ok"
        }

    except Exception as e:
        return {
            "sampling_rate": np.nan,
            "duration": np.nan,
            "rms": np.nan,
            "peak": np.nan,
            "zcr": np.nan,
            "spectral_centroid": np.nan,
            "spectral_bandwidth": np.nan,
            "spectral_rolloff": np.nan,
            "status": f"error: {e}"
        }

def summarize(values):
    values = pd.to_numeric(
        values,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan
        }

    return {
        "count": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "median": values.median(),
        "min": values.min(),
        "max": values.max()
    }

def main():
    print("Loading metadata")

    frames = []

    for split, path in METADATA_FILES.items():
        if not os.path.exists(path):
            print(
                f"ERROR: Missing metadata: {path}"
            )
            sys.exit(1)

        df = pd.read_csv(path)
        df["split"] = split

        frames.append(df)

        print(
            f"{split}: {len(df)} samples"
        )

    metadata = pd.concat(
        frames,
        ignore_index=True
    )

    icbhi_copd = metadata[
        (metadata["source"] == "ICBHI") &
        (metadata["class_name"] == "COPD") &
        (metadata["device"] if "device" in metadata.columns else metadata["audio_path"].apply(extract_device) == "LittC2SE")
    ].copy()

    if "device" not in icbhi_copd.columns:
        icbhi_copd["device"] = icbhi_copd[
            "audio_path"
        ].apply(
            extract_device
        )

    icbhi_copd = icbhi_copd[
        icbhi_copd["device"] == "LittC2SE"
    ].copy()

    target = icbhi_copd[
        icbhi_copd["patient_group"].isin(
            TARGET_PATIENTS
        )
    ].copy()

    print()
    print(
        "LittC2SE COPD target patients"
    )

    print(
        target.groupby(
            [
                "patient_group",
                "split"
            ]
        ).size().to_string()
    )

    if len(target) == 0:
        print(
            "ERROR: No target samples found."
        )
        sys.exit(1)

    print()
    print("Target patient metadata")

    print(
        target[
            [
                "patient_group",
                "split",
                "audio_path"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "patient_group",
                "audio_path"
            ]
        )
        .to_string(index=False)
    )

    audio_paths = sorted(
        target["audio_path"]
        .dropna()
        .unique()
    )

    print()
    print(
        f"Unique target audio files: "
        f"{len(audio_paths)}"
    )

    audio_rows = []

    for path in tqdm(
        audio_paths,
        desc="Analyzing target recordings",
        unit="file"
    ):
        features = audio_features(path)

        rows = target[
            target["audio_path"] == path
        ]

        audio_rows.append({
            "patient_group": rows[
                "patient_group"
            ].iloc[0],
            "split": rows[
                "split"
            ].iloc[0],
            "audio_path": path,
            "cycle_count": len(rows),
            **features
        })

    audio_df = pd.DataFrame(
        audio_rows
    )

    target = target.copy()

    target["cycle_duration"] = (
        pd.to_numeric(
            target["end_time"],
            errors="coerce"
        ) -
        pd.to_numeric(
            target["start_time"],
            errors="coerce"
        )
    )

    target = target.merge(
        audio_df,
        on=[
            "patient_group",
            "split",
            "audio_path"
        ],
        how="left",
        suffixes=(
            "",
            "_audio"
        )
    )

    print()
    print("=" * 70)
    print("PATIENT LEVEL SUMMARY")
    print("=" * 70)

    patient_rows = []

    numeric_cycle_features = [
        "cycle_duration",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff"
    ]

    for patient, group in target.groupby(
        "patient_group"
    ):
        row = {
            "patient_group": patient,
            "split": group["split"].iloc[0],
            "cycles": len(group),
            "audio_files": group[
                "audio_path"
            ].nunique()
        }

        for feature in numeric_cycle_features:
            values = pd.to_numeric(
                group[feature],
                errors="coerce"
            ).dropna()

            row[f"{feature}_mean"] = (
                values.mean()
                if len(values) > 0
                else np.nan
            )

            row[f"{feature}_median"] = (
                values.median()
                if len(values) > 0
                else np.nan
            )

        row["sampling_rate"] = (
            group["sampling_rate"]
            .mode()
            .iloc[0]
            if len(
                group["sampling_rate"]
                .dropna()
            ) > 0
            else np.nan
        )

        if "sound_class" in group.columns:
            sound_counts = (
                group["sound_class"]
                .value_counts()
                .to_dict()
            )

            row["sound_class_distribution"] = str(
                sound_counts
            )

        patient_rows.append(row)

    patient_summary = pd.DataFrame(
        patient_rows
    )

    print(
        patient_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    patient_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_patient_summary.csv"
        ),
        index=False
    )

    print()
    print("=" * 70)
    print("AUDIO FILE LEVEL SUMMARY")
    print("=" * 70)

    audio_summary = (
        audio_df
        .sort_values(
            [
                "patient_group",
                "audio_path"
            ]
        )
    )

    print(
        audio_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    audio_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_audio_features.csv"
        ),
        index=False
    )

    print()
    print("=" * 70)
    print("SOUND CLASS DISTRIBUTION")
    print("=" * 70)

    if "sound_class" in target.columns:
        sound_distribution = pd.crosstab(
            target["patient_group"],
            target["sound_class"]
        )

        print(
            sound_distribution.to_string()
        )

        sound_distribution.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "littc2se_copd_sound_class_distribution.csv"
            )
        )

    print()
    print("=" * 70)
    print("PATIENT FEATURE COMPARISON")
    print("=" * 70)

    comparison_features = [
        "cycle_duration_mean",
        "rms_mean",
        "peak_mean",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean"
    ]

    comparison = patient_summary[
        [
            "patient_group",
            "split"
        ] + comparison_features
    ].copy()

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_feature_comparison.csv"
        ),
        index=False
    )

    print()
    print("=" * 70)
    print("PAIRWISE DIFFERENCES")
    print("=" * 70)

    patient_lookup = patient_summary.set_index(
        "patient_group"
    )

    pair_rows = []

    pairs = [
        (
            "ICBHI_134",
            "ICBHI_141"
        ),
        (
            "ICBHI_199",
            "ICBHI_141"
        ),
        (
            "ICBHI_134",
            "ICBHI_199"
        )
    ]

    for patient_a, patient_b in pairs:
        row_a = patient_lookup.loc[
            patient_a
        ]

        row_b = patient_lookup.loc[
            patient_b
        ]

        for feature in comparison_features:
            value_a = row_a[feature]
            value_b = row_b[feature]

            pair_rows.append({
                "patient_a": patient_a,
                "patient_b": patient_b,
                "feature": feature,
                "patient_a_value": value_a,
                "patient_b_value": value_b,
                "absolute_difference": (
                    value_a - value_b
                ),
                "percent_difference": (
                    ((value_a - value_b) / value_b) * 100
                    if value_b != 0
                    else np.nan
                )
            })

    pairwise = pd.DataFrame(
        pair_rows
    )

    print(
        pairwise.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    pairwise.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_pairwise_differences.csv"
        ),
        index=False
    )

    print()
    print("=" * 70)
    print("OUTPUT DIRECTORY")
    print("=" * 70)

    print(OUTPUT_DIR)

    print()
    print("Generated files:")

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):
        print(filename)

if __name__ == "__main__":
    main()