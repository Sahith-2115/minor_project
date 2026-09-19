from pathlib import Path
import pandas as pd
import soundfile as sf
import librosa
import numpy as np

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "unified_dataset"

SPLITS = ["train", "validation", "test"]

def inspect_file(path):
    try:
        info = sf.info(str(path))
        original_sr = info.samplerate
        channels = info.channels
        frames = info.frames
        duration = frames / original_sr if original_sr else 0.0

        y, sr = librosa.load(str(path), sr=16000, mono=True)

        rms = float(np.sqrt(np.mean(y ** 2)))
        peak = float(np.max(np.abs(y))) if len(y) else 0.0

        return {
            "original_sr": original_sr,
            "channels": channels,
            "duration": duration,
            "resampled_sr": sr,
            "resampled_samples": len(y),
            "rms": rms,
            "peak": peak,
            "status": "OK"
        }
    except Exception as e:
        return {
            "original_sr": np.nan,
            "channels": np.nan,
            "duration": np.nan,
            "resampled_sr": np.nan,
            "resampled_samples": np.nan,
            "rms": np.nan,
            "peak": np.nan,
            "status": f"ERROR: {e}"
        }

def main():
    all_results = []

    for split in SPLITS:
        metadata_path = METADATA_DIR / f"{split}_metadata.csv"

        if not metadata_path.exists():
            print(f"Missing metadata: {metadata_path}")
            continue

        df = pd.read_csv(metadata_path)

        bronchial = df[df["class_name"] == "Bronchial"].copy()

        print()
        print("=" * 80)
        print(f"{split.upper()} BRONCHIAL")
        print("=" * 80)
        print(f"Recordings: {len(bronchial)}")
        print(f"Patient groups: {bronchial['patient_group'].nunique()}")

        split_results = []

        for _, row in bronchial.iterrows():
            audio_path = Path(row["audio_path"])

            result = inspect_file(audio_path)

            result["split"] = split
            result["audio_path"] = str(audio_path)
            result["patient_group"] = row["patient_group"]
            result["class_name"] = row["class_name"]

            split_results.append(result)
            all_results.append(result)

        split_df = pd.DataFrame(split_results)

        print()
        print("Original sampling rate:")
        print(split_df["original_sr"].value_counts().sort_index())

        print()
        print("Original sampling rate percentages:")
        print(
            (
                split_df["original_sr"]
                .value_counts(normalize=True)
                .sort_index()
                * 100
            ).round(2)
        )

        print()
        print("Duration statistics:")
        print(
            split_df["duration"]
            .describe()[
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ].round(4)
        )

        print()
        print("RMS statistics:")
        print(
            split_df["rms"]
            .describe()[
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ].round(6)
        )

        print()
        print("Peak statistics:")
        print(
            split_df["peak"]
            .describe()[
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ].round(6)
        )

        print()
        print("Sampling rate by patient group:")

        patient_sr = (
            split_df.groupby(["patient_group", "original_sr"])
            .size()
            .reset_index(name="recordings")
            .sort_values(["patient_group", "original_sr"])
        )

        print(patient_sr.to_string(index=False))

    results_df = pd.DataFrame(all_results)

    output_dir = PROJECT_ROOT / "outputs" / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "bronchial_sampling_rate_audit.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print("=" * 80)
    print("OVERALL BRONCHIAL SAMPLING RATE")
    print("=" * 80)

    print(
        results_df["original_sr"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Overall percentages:")

    print(
        (
            results_df["original_sr"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2).to_string()
    )

    print()
    print(f"Audit saved to:")
    print(output_path)

if __name__ == "__main__":
    main()