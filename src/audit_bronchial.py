import os
import warnings
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "unified_dataset"
)

SPLITS = {
    "train": os.path.join(DATASET_DIR, "train_metadata.csv"),
    "validation": os.path.join(DATASET_DIR, "validation_metadata.csv"),
    "test": os.path.join(DATASET_DIR, "test_metadata.csv")
}

TARGET_CLASS = "Bronchial"

SAMPLE_RATE = 16000

FEATURE_COLUMNS = [
    "duration",
    "rms",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "peak_amplitude"
]

def get_audio_path(row):
    path = str(row["audio_path"])

    if os.path.isabs(path):
        return path

    return os.path.join(PROJECT_ROOT, path)

def extract_features(audio_path):
    try:
        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(y) == 0:
            return None

        duration = len(y) / sr

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(y)
                )
            )
        )

        zcr = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    y
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
                    sr=sr
                )
            )
        )

        peak_amplitude = float(
            np.max(
                np.abs(y)
            )
        )

        return {
            "duration": duration,
            "rms": rms,
            "zcr": zcr,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "peak_amplitude": peak_amplitude,
            "original_sr": sr
        }

    except Exception as e:
        print(
            f"ERROR: {audio_path}\n"
            f"{e}"
        )
        return None

def summarize_split(df):
    summary = {}

    for column in FEATURE_COLUMNS:
        values = df[column].dropna()

        if len(values) == 0:
            continue

        summary[column] = {
            "count": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "25%": values.quantile(0.25),
            "median": values.median(),
            "75%": values.quantile(0.75),
            "max": values.max()
        }

    return summary

def print_summary(summary, split_name):
    print()
    print("=" * 90)
    print(f"{split_name.upper()} BRONCHIAL FEATURE SUMMARY")
    print("=" * 90)

    for feature, values in summary.items():
        print()
        print(f"{feature}:")
        print(
            f"  Mean   : {values['mean']:.6f}"
        )
        print(
            f"  Std    : {values['std']:.6f}"
        )
        print(
            f"  Min    : {values['min']:.6f}"
        )
        print(
            f"  25%    : {values['25%']:.6f}"
        )
        print(
            f"  Median : {values['median']:.6f}"
        )
        print(
            f"  75%    : {values['75%']:.6f}"
        )
        print(
            f"  Max    : {values['max']:.6f}"
        )

def print_comparison(stats):
    print()
    print("=" * 110)
    print("TRAIN VS VALIDATION VS TEST")
    print("=" * 110)

    rows = []

    for feature in FEATURE_COLUMNS:
        row = {
            "feature": feature
        }

        for split in ["train", "validation", "test"]:
            if feature in stats[split]:
                row[f"{split}_mean"] = stats[split][feature]["mean"]
                row[f"{split}_std"] = stats[split][feature]["std"]
            else:
                row[f"{split}_mean"] = np.nan
                row[f"{split}_std"] = np.nan

        rows.append(row)

    comparison = pd.DataFrame(rows)

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        200
    )

    pd.set_option(
        "display.float_format",
        lambda x: f"{x:.6f}"
    )

    print(comparison.to_string(index=False))

def calculate_distribution_difference(stats):
    print()
    print("=" * 90)
    print("TEST DISTRIBUTION DIFFERENCE FROM TRAIN")
    print("=" * 90)

    rows = []

    for feature in FEATURE_COLUMNS:
        train_mean = stats["train"][feature]["mean"]
        train_std = stats["train"][feature]["std"]
        test_mean = stats["test"][feature]["mean"]
        test_std = stats["test"][feature]["std"]

        if train_std > 0:
            mean_shift = (
                test_mean - train_mean
            ) / train_std
        else:
            mean_shift = np.nan

        rows.append({
            "feature": feature,
            "train_mean": train_mean,
            "test_mean": test_mean,
            "train_std": train_std,
            "test_std": test_std,
            "test_mean_shift_in_train_std": mean_shift
        })

    df = pd.DataFrame(rows)

    pd.set_option(
        "display.float_format",
        lambda x: f"{x:.6f}"
    )

    print(df.to_string(index=False))

def analyze_patients(metadata):
    print()
    print("=" * 90)
    print("PATIENT LEVEL ANALYSIS")
    print("=" * 90)

    for split, df in metadata.items():
        patients = df["patient_group"].nunique()

        print()
        print(
            f"{split.upper()}: "
            f"{len(df)} recordings, "
            f"{patients} patient groups"
        )

        counts = (
            df.groupby("patient_group")
            .size()
            .sort_values(ascending=False)
        )

        print(
            "Recordings per patient:"
        )

        print(
            counts.describe().to_string()
        )

def analyze_sources(metadata):
    print()
    print("=" * 90)
    print("SOURCE ANALYSIS")
    print("=" * 90)

    for split, df in metadata.items():
        print()
        print(split.upper())

        if "source" in df.columns:
            print(
                df["source"]
                .value_counts()
                .to_string()
            )

        if "audio_path" in df.columns:
            source_counts = []

            for path in df["audio_path"]:
                path = str(path)

                if "Asthma Detection Dataset" in path:
                    source = "Asthma_V2"
                elif "audio_and_text_files" in path:
                    source = "ICBHI"
                else:
                    source = "Other"

                source_counts.append(source)

            print()
            print(
                pd.Series(source_counts)
                .value_counts()
                .to_string()
            )

def save_results(all_results):
    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "bronchial_audit"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    combined = []

    for split, df in all_results.items():
        temp = df.copy()
        temp.insert(
            0,
            "split",
            split
        )
        combined.append(temp)

    combined_df = pd.concat(
        combined,
        ignore_index=True
    )

    output_path = os.path.join(
        output_dir,
        "bronchial_audio_features.csv"
    )

    combined_df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Feature data saved to:\n{output_path}"
    )

def main():
    print("=" * 90)
    print("BRONCHIAL DATA AUDIT")
    print("=" * 90)
    print(
        f"Target class: {TARGET_CLASS}"
    )
    print(
        f"Sample rate for analysis: {SAMPLE_RATE} Hz"
    )

    metadata = {}
    all_results = {}

    for split, metadata_path in SPLITS.items():
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found:\n{metadata_path}"
            )

        df = pd.read_csv(
            metadata_path
        )

        df = df[
            df["class_name"] == TARGET_CLASS
        ].copy()

        metadata[split] = df

        print()
        print(
            f"{split.upper()}: "
            f"{len(df)} Bronchial recordings"
        )

    analyze_patients(metadata)
    analyze_sources(metadata)

    for split, df in metadata.items():
        print()
        print("=" * 90)
        print(
            f"EXTRACTING FEATURES: {split.upper()}"
        )
        print("=" * 90)

        rows = []

        for index, row in df.iterrows():
            audio_path = get_audio_path(
                row
            )

            if not os.path.exists(audio_path):
                print(
                    f"MISSING: {audio_path}"
                )
                continue

            features = extract_features(
                audio_path
            )

            if features is None:
                continue

            result = {
                "split": split,
                "audio_path": audio_path,
                "patient_group": row[
                    "patient_group"
                ],
                "class_name": row[
                    "class_name"
                ]
            }

            for key, value in features.items():
                result[key] = value

            rows.append(result)

            if len(rows) % 20 == 0:
                print(
                    f"Processed {len(rows)}/{len(df)}"
                )

        result_df = pd.DataFrame(
            rows
        )

        all_results[split] = result_df

        print(
            f"Successfully processed: "
            f"{len(result_df)}/{len(df)}"
        )

    stats = {}

    for split, df in all_results.items():
        stats[split] = summarize_split(
            df
        )

        print_summary(
            stats[split],
            split
        )

    print_comparison(
        stats
    )

    calculate_distribution_difference(
        stats
    )

    save_results(
        all_results
    )

    print()
    print("=" * 90)
    print("BRONCHIAL AUDIT COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    main()