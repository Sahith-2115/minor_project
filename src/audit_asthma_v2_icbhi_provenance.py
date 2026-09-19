from pathlib import Path
import hashlib
import re
import pandas as pd
import librosa
import numpy as np

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
ICBHI_DIR = PROJECT_ROOT / "data/raw/audio_and_text_files"
ASTHMA_V2_DIR = PROJECT_ROOT / "data/raw/asthma_detection_v2/Asthma Detection Dataset Version 2"
OUTPUT_DIR = PROJECT_ROOT / "outputs/provenance_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["Asthma", "Bronchial", "COPD", "Healthy", "Pneumonia"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalized_name(name):
    name = Path(name).stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def extract_icbhi_id(filename):
    stem = Path(filename).stem
    m = re.match(r"^(\d+)_", stem)
    if m:
        return m.group(1)
    return ""


def extract_local_patient_id(filename):
    stem = Path(filename).stem
    patterns = [
        r"(?i)(?:patient|subject|person|participant)[_\- ]?(\d+)",
        r"(?i)^p[_\- ]?(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, stem)
        if m:
            return m.group(1)
    return ""


def audio_signature(path):
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)

        if len(y) == 0:
            return None

        duration = len(y) / sr
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
        peak = float(np.max(np.abs(y)))

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr
                )[0]
            )
        )

        bandwidth = float(
            np.mean(
                librosa.feature.spectral_bandwidth(
                    y=y,
                    sr=sr
                )[0]
            )
        )

        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=y,
                    sr=sr,
                    roll_percent=0.85
                )[0]
            )
        )

        return {
            "duration": duration,
            "rms": rms,
            "peak": peak,
            "zcr": zcr,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "sample_rate_after_resampling": sr,
        }

    except Exception:
        return None


def collect_icbhi():
    rows = []

    wav_files = sorted(ICBHI_DIR.glob("*.wav"))

    print("=" * 70)
    print("ICBHI FILE AUDIT")
    print("=" * 70)
    print(f"ICBHI WAV files found: {len(wav_files)}")

    for path in wav_files:
        stem = path.stem

        patient_id = extract_icbhi_id(path.name)

        rows.append(
            {
                "file": path.name,
                "path": str(path),
                "normalized_name": normalized_name(path.name),
                "patient_id": patient_id,
                "sha256": sha256_file(path),
            }
        )

    return pd.DataFrame(rows)


def collect_asthma_v2():
    rows = []

    print()
    print("=" * 70)
    print("ASTHMA V2 FILE AUDIT")
    print("=" * 70)

    for class_dir in sorted(ASTHMA_V2_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        wav_files = sorted(class_dir.glob("*.wav"))

        print(f"{class_name:12s}: {len(wav_files)} WAV files")

        for path in wav_files:
            rows.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "source_class": class_name,
                    "normalized_name": normalized_name(path.name),
                    "local_patient_id": extract_local_patient_id(path.name),
                    "sha256": sha256_file(path),
                }
            )

    return pd.DataFrame(rows)


def exact_hash_overlap(icbhi, asthma):
    print()
    print("=" * 70)
    print("EXACT SHA256 OVERLAP")
    print("=" * 70)

    icbhi_hashes = set(icbhi["sha256"])
    asthma_hashes = set(asthma["sha256"])

    overlap = icbhi_hashes.intersection(asthma_hashes)

    print(f"ICBHI unique hashes: {len(icbhi_hashes)}")
    print(f"Asthma V2 unique hashes: {len(asthma_hashes)}")
    print(f"Exact overlapping hashes: {len(overlap)}")

    rows = []

    if overlap:
        for h in sorted(overlap):
            icbhi_files = icbhi.loc[icbhi["sha256"] == h, "file"].tolist()
            asthma_files = asthma.loc[asthma["sha256"] == h, "file"].tolist()

            rows.append(
                {
                    "sha256": h,
                    "icbhi_files": "|".join(icbhi_files),
                    "asthma_v2_files": "|".join(asthma_files),
                }
            )

            print("OVERLAP")
            print("ICBHI:", icbhi_files)
            print("Asthma V2:", asthma_files)

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "exact_hash_overlap.csv",
        index=False
    )

    return overlap


def filename_overlap(icbhi, asthma):
    print()
    print("=" * 70)
    print("NORMALIZED FILENAME OVERLAP")
    print("=" * 70)

    icbhi_map = {}

    for _, row in icbhi.iterrows():
        icbhi_map.setdefault(row["normalized_name"], []).append(row["file"])

    asthma_map = {}

    for _, row in asthma.iterrows():
        asthma_map.setdefault(row["normalized_name"], []).append(row["file"])

    common = sorted(set(icbhi_map).intersection(asthma_map))

    print(f"Common normalized filenames: {len(common)}")

    rows = []

    for key in common:
        rows.append(
            {
                "normalized_name": key,
                "icbhi_files": "|".join(icbhi_map[key]),
                "asthma_v2_files": "|".join(asthma_map[key]),
            }
        )

        print()
        print("Normalized match:", key)
        print("ICBHI:", icbhi_map[key])
        print("Asthma V2:", asthma_map[key])

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "normalized_filename_overlap.csv",
        index=False
    )

    return common


def inspect_filenames(asthma):
    print()
    print("=" * 70)
    print("ASTHMA V2 FILENAME PATTERNS")
    print("=" * 70)

    for class_name in TARGET_CLASSES:
        subset = asthma[asthma["source_class"] == class_name]

        print()
        print(f"[{class_name}]")
        print(f"Files: {len(subset)}")

        for filename in subset["file"].head(20):
            print(filename)

        if len(subset) > 20:
            print(f"... and {len(subset) - 20} more")


def audio_statistics(asthma):
    print()
    print("=" * 70)
    print("ASTHMA V2 AUDIO STATISTICS")
    print("=" * 70)

    rows = []

    total = len(asthma)

    for index, (_, row) in enumerate(asthma.iterrows(), start=1):
        path = Path(row["path"])

        signature = audio_signature(path)

        if signature is None:
            continue

        result = dict(row)
        result.update(signature)
        rows.append(result)

        if index % 100 == 0 or index == total:
            print(f"Processed {index}/{total}")

    df = pd.DataFrame(rows)

    if df.empty:
        print("No audio statistics could be extracted.")
        return df

    output_file = OUTPUT_DIR / "asthma_v2_audio_statistics.csv"
    df.to_csv(output_file, index=False)

    print()
    print(f"Saved: {output_file}")

    summary = (
        df.groupby("source_class")
        [
            [
                "duration",
                "rms",
                "peak",
                "zcr",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
            ]
        ]
        .agg(["count", "mean", "std", "min", "max"])
    )

    print()
    print(summary.to_string())

    summary.to_csv(
        OUTPUT_DIR / "asthma_v2_audio_statistics_summary.csv"
    )

    return df


def compare_icbhi_and_asthma_metadata(icbhi, asthma):
    print()
    print("=" * 70)
    print("PATIENT ID STRUCTURE")
    print("=" * 70)

    print()
    print("ICBHI patient ID examples:")

    print(
        icbhi[
            ["file", "patient_id"]
        ].head(30).to_string(index=False)
    )

    print()
    print("Asthma V2 local patient ID extraction examples:")

    print(
        asthma[
            ["file", "source_class", "local_patient_id"]
        ].head(50).to_string(index=False)
    )

    asthma_with_ids = asthma[asthma["local_patient_id"] != ""]

    print()
    print(
        "Asthma V2 files with recognizable patient ID pattern:",
        len(asthma_with_ids),
        "/",
        len(asthma)
    )

    patient_summary = (
        asthma_with_ids
        .groupby(["source_class", "local_patient_id"])
        .size()
        .reset_index(name="recording_count")
        .sort_values(
            ["source_class", "local_patient_id"]
        )
    )

    patient_summary.to_csv(
        OUTPUT_DIR / "asthma_v2_patient_id_summary.csv",
        index=False
    )

    print()
    print("Patient group summary:")
    print(patient_summary.to_string(index=False))


def main():
    print()
    print("=" * 70)
    print("ASTHMA V2 <-> ICBHI PROVENANCE AUDIT")
    print("=" * 70)
    print()

    if not ICBHI_DIR.exists():
        raise FileNotFoundError(
            f"ICBHI directory not found: {ICBHI_DIR}"
        )

    if not ASTHMA_V2_DIR.exists():
        raise FileNotFoundError(
            f"Asthma V2 directory not found: {ASTHMA_V2_DIR}"
        )

    icbhi = collect_icbhi()
    asthma = collect_asthma_v2()

    icbhi.to_csv(
        OUTPUT_DIR / "icbhi_file_inventory.csv",
        index=False
    )

    asthma.to_csv(
        OUTPUT_DIR / "asthma_v2_file_inventory.csv",
        index=False
    )

    exact_hash_overlap(
        icbhi,
        asthma
    )

    filename_overlap(
        icbhi,
        asthma
    )

    inspect_filenames(
        asthma
    )

    compare_icbhi_and_asthma_metadata(
        icbhi,
        asthma
    )

    audio_statistics(
        asthma
    )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print()
    print("Reports saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()