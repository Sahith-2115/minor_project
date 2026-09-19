import os
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"
TEST_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/test_metadata.csv"
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit"
)

PATIENT_ID = "ICBHI_134"
SOURCE = "ICBHI"
DISEASE = "COPD"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def audio_features(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)

        if len(y) == 0:
            return {
                "sampling_rate": sr,
                "audio_duration": 0.0,
                "rms": np.nan,
                "peak": np.nan,
                "zcr": np.nan,
                "spectral_centroid": np.nan,
                "spectral_bandwidth": np.nan,
                "spectral_rolloff": np.nan,
                "status": "empty"
            }

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
                    sr=sr,
                    roll_percent=0.85
                )
            )
        )

        return {
            "sampling_rate": sr,
            "audio_duration": float(len(y) / sr),
            "rms": rms,
            "peak": peak,
            "zcr": zcr,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "status": "ok"
        }

    except Exception as e:
        return {
            "sampling_rate": np.nan,
            "audio_duration": np.nan,
            "rms": np.nan,
            "peak": np.nan,
            "zcr": np.nan,
            "spectral_centroid": np.nan,
            "spectral_bandwidth": np.nan,
            "spectral_rolloff": np.nan,
            "status": f"error: {e}"
        }

def percentile_rank(value, reference):
    reference = pd.Series(reference).dropna()

    if len(reference) == 0 or pd.isna(value):
        return np.nan

    return float((reference <= value).mean() * 100)

def summarize_group(df, name):
    numeric_cols = [
        "cycle_duration",
        "audio_duration",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff"
    ]

    rows = []

    for col in numeric_cols:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(values) == 0:
            continue

        rows.append({
            "group": name,
            "feature": col,
            "count": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
            "p10": values.quantile(0.10),
            "p25": values.quantile(0.25),
            "p75": values.quantile(0.75),
            "p90": values.quantile(0.90)
        })

    return pd.DataFrame(rows)

def main():
    print(f"Loading test metadata from: {TEST_METADATA}")

    if not os.path.exists(TEST_METADATA):
        print(f"ERROR: Metadata file not found: {TEST_METADATA}")
        sys.exit(1)

    metadata = pd.read_csv(TEST_METADATA)

    required_columns = [
        "sample_id",
        "audio_path",
        "source",
        "class_name",
        "patient_group"
    ]

    missing = [c for c in required_columns if c not in metadata.columns]

    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Available columns: {list(metadata.columns)}")
        sys.exit(1)

    icbhi_copd = metadata[
        (metadata["source"] == SOURCE) &
        (metadata["class_name"] == DISEASE)
    ].copy()

    patient_134 = icbhi_copd[
        icbhi_copd["patient_group"] == PATIENT_ID
    ].copy()

    other_patients = icbhi_copd[
        icbhi_copd["patient_group"] != PATIENT_ID
    ].copy()

    print()
    print("ICBHI COPD test set")
    print(f"Total cycles: {len(icbhi_copd)}")
    print(f"Patient 134 cycles: {len(patient_134)}")
    print(f"Other COPD cycles: {len(other_patients)}")
    print(f"Other COPD patient groups: {other_patients['patient_group'].nunique()}")

    if len(patient_134) == 0:
        print(f"ERROR: No samples found for {PATIENT_ID}")
        sys.exit(1)

    all_audio_paths = sorted(
        icbhi_copd["audio_path"].dropna().unique()
    )

    print()
    print(f"Unique ICBHI COPD audio files: {len(all_audio_paths)}")

    audio_records = []

    for path in tqdm(
        all_audio_paths,
        desc="Analyzing ICBHI COPD audio files",
        unit="file"
    ):
        features = audio_features(path)

        rows = icbhi_copd[
            icbhi_copd["audio_path"] == path
        ]

        patient_groups = rows["patient_group"].unique()

        audio_records.append({
            "audio_path": path,
            "patient_group": "|".join(patient_groups),
            "cycle_count": len(rows),
            **features
        })

    audio_df = pd.DataFrame(audio_records)

    cycle_df = icbhi_copd.copy()

    cycle_df["cycle_duration"] = (
        pd.to_numeric(cycle_df.get("end_time"), errors="coerce") -
        pd.to_numeric(cycle_df.get("start_time"), errors="coerce")
    )

    cycle_df = cycle_df.merge(
        audio_df,
        on="audio_path",
        how="left",
        suffixes=("", "_audio")
    )

    cycle_df["is_patient_134"] = (
        cycle_df["patient_group"] == PATIENT_ID
    )

    patient_134_cycles = cycle_df[
        cycle_df["is_patient_134"]
    ].copy()

    other_cycles = cycle_df[
        ~cycle_df["is_patient_134"]
    ].copy()

    patient_134_audio_paths = set(
        patient_134_cycles["audio_path"].dropna()
    )

    other_audio_paths = set(
        other_cycles["audio_path"].dropna()
    )

    patient_134_audio = audio_df[
        audio_df["audio_path"].isin(patient_134_audio_paths)
    ].copy()

    other_audio = audio_df[
        audio_df["audio_path"].isin(other_audio_paths)
    ].copy()

    print()
    print("Patient 134 audio")
    print(f"Unique audio files: {len(patient_134_audio)}")

    print()
    print("Patient 134 cycle sound classes")

    if "sound_class" in patient_134_cycles.columns:
        print(
            patient_134_cycles["sound_class"]
            .value_counts()
            .to_string()
        )

    print()
    print("Patient 134 sampling rates")

    print(
        patient_134_audio["sampling_rate"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    cycle_summary_134 = summarize_group(
        patient_134_cycles,
        PATIENT_ID
    )

    cycle_summary_other = summarize_group(
        other_cycles,
        "Other ICBHI COPD test patients"
    )

    cycle_summary = pd.concat(
        [
            cycle_summary_134,
            cycle_summary_other
        ],
        ignore_index=True
    )

    audio_summary_134 = summarize_group(
        patient_134_audio,
        f"{PATIENT_ID} audio files"
    )

    audio_summary_other = summarize_group(
        other_audio,
        "Other ICBHI COPD audio files"
    )

    audio_summary = pd.concat(
        [
            audio_summary_134,
            audio_summary_other
        ],
        ignore_index=True
    )

    comparison_rows = []

    features = [
        "cycle_duration",
        "audio_duration",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff"
    ]

    for feature in features:
        p134_values = pd.to_numeric(
            patient_134_cycles[feature],
            errors="coerce"
        ).dropna()

        other_values = pd.to_numeric(
            other_cycles[feature],
            errors="coerce"
        ).dropna()

        if len(p134_values) == 0 or len(other_values) == 0:
            continue

        p134_mean = p134_values.mean()
        other_mean = other_values.mean()
        other_std = other_values.std()

        if other_std > 0:
            z_score = (p134_mean - other_mean) / other_std
        else:
            z_score = np.nan

        comparison_rows.append({
            "feature": feature,
            "patient_134_count": len(p134_values),
            "patient_134_mean": p134_mean,
            "patient_134_median": p134_values.median(),
            "other_patients_count": len(other_values),
            "other_patients_mean": other_mean,
            "other_patients_median": other_values.median(),
            "difference_mean": p134_mean - other_mean,
            "percent_difference": (
                ((p134_mean - other_mean) / other_mean) * 100
                if other_mean != 0 else np.nan
            ),
            "other_distribution_z_score": z_score,
            "patient_134_mean_percentile": percentile_rank(
                p134_mean,
                other_values
            )
        })

    comparison_df = pd.DataFrame(comparison_rows)

    patient_summary_rows = []

    for patient_group, group in icbhi_copd.groupby(
        "patient_group"
    ):
        cycle_group = cycle_df[
            cycle_df["patient_group"] == patient_group
        ]

        audio_paths = cycle_group["audio_path"].dropna().unique()

        audio_group = audio_df[
            audio_df["audio_path"].isin(audio_paths)
        ]

        row = {
            "patient_group": patient_group,
            "cycle_count": len(cycle_group),
            "audio_file_count": len(audio_paths),
            "sound_class_count": (
                cycle_group["sound_class"].nunique()
                if "sound_class" in cycle_group.columns
                else np.nan
            )
        }

        for feature in [
            "cycle_duration",
            "rms",
            "peak",
            "zcr",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff"
        ]:
            values = pd.to_numeric(
                cycle_group[feature],
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

        if len(audio_group) > 0:
            row["sampling_rate"] = (
                audio_group["sampling_rate"]
                .mode()
                .iloc[0]
            )

            row["audio_duration_mean"] = (
                audio_group["audio_duration"].mean()
            )

        else:
            row["sampling_rate"] = np.nan
            row["audio_duration_mean"] = np.nan

        patient_summary_rows.append(row)

    patient_summary = pd.DataFrame(
        patient_summary_rows
    ).sort_values(
        "patient_group"
    )

    patient_134_row = patient_summary[
        patient_summary["patient_group"] == PATIENT_ID
    ]

    other_patient_summary = patient_summary[
        patient_summary["patient_group"] != PATIENT_ID
    ].copy()

    for feature in [
        "cycle_duration_mean",
        "rms_mean",
        "peak_mean",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean"
    ]:
        if (
            len(patient_134_row) > 0 and
            feature in patient_summary.columns
        ):
            value = patient_134_row[feature].iloc[0]

            reference = other_patient_summary[
                feature
            ].dropna()

            patient_summary.loc[
                patient_summary["patient_group"] == PATIENT_ID,
                f"{feature}_patient_percentile"
            ] = percentile_rank(
                value,
                reference
            )

    cycle_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icbhi_copd_test_cycle_level_features.csv"
        ),
        index=False
    )

    audio_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icbhi_copd_test_audio_level_features.csv"
        ),
        index=False
    )

    cycle_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "patient_134_vs_other_cycle_summary.csv"
        ),
        index=False
    )

    audio_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "patient_134_vs_other_audio_summary.csv"
        ),
        index=False
    )

    comparison_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "patient_134_feature_comparison.csv"
        ),
        index=False
    )

    patient_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icbhi_copd_test_patient_summary.csv"
        ),
        index=False
    )

    patient_134_cycles.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "patient_134_cycles.csv"
        ),
        index=False
    )

    print()
    print("Patient 134 vs other ICBHI COPD test patients")
    print()

    if len(comparison_df) > 0:
        display_cols = [
            "feature",
            "patient_134_mean",
            "other_patients_mean",
            "difference_mean",
            "percent_difference",
            "other_distribution_z_score",
            "patient_134_mean_percentile"
        ]

        print(
            comparison_df[display_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    print()
    print("Patient 134 patient level summary")

    if len(patient_134_row) > 0:
        print(
            patient_134_row.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("Saved files:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(filename)

if __name__ == "__main__":
    main()