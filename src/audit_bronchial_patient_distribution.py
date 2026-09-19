from pathlib import Path
import pandas as pd
import librosa
import numpy as np

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "unified_dataset"

SPLITS = ["train", "validation", "test"]

def extract_features(path):
    y, sr = librosa.load(str(path), sr=16000, mono=True)

    if len(y) == 0:
        return {
            "rms": np.nan,
            "peak": np.nan,
            "zcr": np.nan,
            "centroid": np.nan,
            "bandwidth": np.nan,
            "rolloff": np.nan
        }

    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

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
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff
    }

def main():
    results = []

    for split in SPLITS:
        metadata_path = METADATA_DIR / f"{split}_metadata.csv"

        df = pd.read_csv(metadata_path)

        df = df[df["class_name"] == "Bronchial"].copy()

        for _, row in df.iterrows():
            features = extract_features(row["audio_path"])

            results.append({
                "split": split,
                "patient_group": row["patient_group"],
                "audio_path": row["audio_path"],
                **features
            })

    result_df = pd.DataFrame(results)

    patient_summary = (
        result_df
        .groupby(["split", "patient_group"])
        .agg(
            recordings=("audio_path", "count"),
            rms_mean=("rms", "mean"),
            rms_std=("rms", "std"),
            peak_mean=("peak", "mean"),
            zcr_mean=("zcr", "mean"),
            centroid_mean=("centroid", "mean"),
            bandwidth_mean=("bandwidth", "mean"),
            rolloff_mean=("rolloff", "mean")
        )
        .reset_index()
    )

    print()
    print("=" * 100)
    print("BRONCHIAL PATIENT LEVEL ACOUSTIC DISTRIBUTION")
    print("=" * 100)

    for split in SPLITS:
        print()
        print("=" * 100)
        print(split.upper())
        print("=" * 100)

        subset = patient_summary[
            patient_summary["split"] == split
        ].copy()

        print(
            subset[
                [
                    "patient_group",
                    "recordings",
                    "rms_mean",
                    "peak_mean",
                    "zcr_mean",
                    "centroid_mean",
                    "bandwidth_mean",
                    "rolloff_mean"
                ]
            ].round(4).to_string(index=False)
        )

    print()
    print("=" * 100)
    print("SPLIT LEVEL PATIENT DISTRIBUTION")
    print("=" * 100)

    split_summary = (
        patient_summary
        .groupby("split")
        .agg(
            patients=("patient_group", "count"),
            recordings=("recordings", "sum"),
            rms_mean=("rms_mean", "mean"),
            rms_std=("rms_mean", "std"),
            peak_mean=("peak_mean", "mean"),
            centroid_mean=("centroid_mean", "mean"),
            bandwidth_mean=("bandwidth_mean", "mean"),
            rolloff_mean=("rolloff_mean", "mean")
        )
        .reset_index()
    )

    print(split_summary.round(4).to_string(index=False))

    output_dir = PROJECT_ROOT / "outputs" / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "bronchial_patient_distribution_audit.csv"

    patient_summary.to_csv(output_path, index=False)

    print()
    print(f"Audit saved to:")
    print(output_path)

if __name__ == "__main__":
    main()