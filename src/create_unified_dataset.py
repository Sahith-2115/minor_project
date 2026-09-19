from pathlib import Path
import hashlib
import random
import re
import numpy as np
import pandas as pd

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
ICBHI_DIR = PROJECT_ROOT / "data/raw/audio_and_text_files"
DIAGNOSIS_FILE = PROJECT_ROOT / "data/raw/patient_diagnosis.csv"
ASTHMA_V2_DIR = PROJECT_ROOT / "data/raw/asthma_detection_v2/Asthma Detection Dataset Version 2"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/unified_dataset"
REPORT_DIR = OUTPUT_DIR / "reports"

UNIFIED_METADATA_FILE = OUTPUT_DIR / "unified_dataset_metadata.csv"
TRAIN_METADATA_FILE = OUTPUT_DIR / "train_metadata.csv"
VAL_METADATA_FILE = OUTPUT_DIR / "validation_metadata.csv"
TEST_METADATA_FILE = OUTPUT_DIR / "test_metadata.csv"

CLASSES = [
    "Asthma",
    "Bronchitis",
    "COPD",
    "Healthy",
    "Pneumonia"
]

ICBHI_TARGET_DIAGNOSES = {
    "COPD",
    "Healthy",
    "Pneumonia"
}

ASTHMA_V2_CLASS_MAP = {
    "asthma": "Asthma",
    "Bronchial": "Bronchitis",
    "copd": "COPD",
    "healthy": "Healthy",
    "pneumonia": "Pneumonia"
}


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def extract_icbhi_patient_id(filename):
    match = re.match(r"^(\d+)_", filename)

    if match:
        return match.group(1)

    return None


def extract_asthma_v2_patient_id(filename):
    match = re.search(r"^P(\d+)", filename, re.IGNORECASE)

    if match:
        return f"P{match.group(1)}"

    return None


def load_diagnosis():
    df = pd.read_csv(DIAGNOSIS_FILE)

    df["ID"] = df["ID"].astype(str).str.strip()
    df["Symptom"] = df["Symptom"].astype(str).str.strip()

    return dict(zip(df["ID"], df["Symptom"]))


def collect_icbhi_cycles():
    diagnosis = load_diagnosis()
    rows = []

    annotation_files = sorted(ICBHI_DIR.glob("*.txt"))

    print(f"ICBHI annotation files found: {len(annotation_files)}")

    for annotation_file in annotation_files:
        patient_id = extract_icbhi_patient_id(annotation_file.name)

        if patient_id is None:
            continue

        diagnosis_label = diagnosis.get(patient_id)

        if diagnosis_label not in ICBHI_TARGET_DIAGNOSES:
            continue

        wav_file = annotation_file.with_suffix(".wav")

        if not wav_file.exists():
            print(f"Warning: missing WAV file for {annotation_file.name}")
            continue

        try:
            with open(annotation_file, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: could not read {annotation_file}: {e}")
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.strip().split()

            if len(parts) < 4:
                continue

            try:
                start_time = float(parts[0])
                end_time = float(parts[1])
            except ValueError:
                continue

            if end_time <= start_time:
                continue

            sound_class = parts[3]

            rows.append({
                "source": "ICBHI",
                "source_class": diagnosis_label,
                "class_name": diagnosis_label,
                "patient_id": patient_id,
                "patient_group": f"ICBHI_{patient_id}",
                "audio_path": str(wav_file),
                "annotation_path": str(annotation_file),
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "sound_class": sound_class,
                "annotation_line": line_number
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["sample_id"] = [
            f"ICBHI_{i:06d}"
            for i in range(len(df))
        ]

    print(f"ICBHI target diagnosis cycles collected: {len(df)}")

    if not df.empty:
        print("\nICBHI class counts:")
        print(
            df["class_name"]
            .value_counts()
            .sort_index()
        )

    return df


def collect_asthma_v2_recordings():
    rows = []

    for source_folder_name, project_class in ASTHMA_V2_CLASS_MAP.items():
        source_folder = ASTHMA_V2_DIR / source_folder_name

        if not source_folder.exists():
            print(f"Warning: folder not found: {source_folder}")
            continue

        wav_files = sorted(source_folder.glob("*.wav"))

        for wav_file in wav_files:
            patient_id = extract_asthma_v2_patient_id(wav_file.name)

            if patient_id is None:
                print(f"Warning: patient ID not found: {wav_file.name}")
                continue

            rows.append({
                "source": "Asthma_V2",
                "source_class": source_folder_name,
                "class_name": project_class,
                "patient_id": patient_id,
                "patient_group": f"AsthmaV2_{source_folder_name}_{patient_id}",
                "audio_path": str(wav_file),
                "annotation_path": "",
                "start_time": 0.0,
                "end_time": np.nan,
                "duration": np.nan,
                "sound_class": "",
                "annotation_line": np.nan
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["sample_id"] = [
            f"AsthmaV2_{i:06d}"
            for i in range(len(df))
        ]

    print(f"\nAsthma V2 recordings collected: {len(df)}")

    if not df.empty:
        print("\nAsthma V2 source class counts:")
        print(
            df["source_class"]
            .value_counts()
            .sort_index()
        )

        print("\nAsthma V2 project class counts:")
        print(
            df["class_name"]
            .value_counts()
            .sort_index()
        )

    return df


def remove_asthma_v2_duplicates(df):
    """
    Remove exact duplicate recordings only from Asthma V2.

    ICBHI must not be deduplicated using WAV hashes because
    multiple respiratory cycles intentionally come from the
    same original ICBHI WAV recording.
    """

    print("\nChecking exact duplicate Asthma V2 recordings...")

    icbhi_df = df[df["source"] == "ICBHI"].copy()
    asthma_df = df[df["source"] == "Asthma_V2"].copy()

    if asthma_df.empty:
        print("No Asthma V2 recordings found.")
        return df.copy()

    hashes = []

    for _, row in asthma_df.iterrows():
        audio_path = Path(row["audio_path"])

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Asthma V2 audio file not found: {audio_path}"
            )

        try:
            file_hash = sha256_file(audio_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to hash {audio_path}: {e}"
            )

        hashes.append(file_hash)

    asthma_df["audio_sha256"] = hashes

    duplicate_mask = asthma_df.duplicated(
        subset=["audio_sha256"],
        keep="first"
    )

    duplicate_count = int(duplicate_mask.sum())

    duplicate_groups = (
        asthma_df[
            asthma_df.duplicated(
                subset=["audio_sha256"],
                keep=False
            )
        ]
        .groupby("audio_sha256")
        .size()
    )

    print(
        f"Asthma V2 recordings before duplicate removal: "
        f"{len(asthma_df)}"
    )

    print(
        f"Asthma V2 exact duplicate recordings found: "
        f"{duplicate_count}"
    )

    print(
        f"Asthma V2 duplicate hash groups: "
        f"{len(duplicate_groups)}"
    )

    asthma_df = asthma_df.loc[
        ~duplicate_mask
    ].copy()

    asthma_df.reset_index(drop=True, inplace=True)

    print(
        f"Asthma V2 recordings after duplicate removal: "
        f"{len(asthma_df)}"
    )

    print(
        f"ICBHI cycles retained without hash deduplication: "
        f"{len(icbhi_df)}"
    )

    result = pd.concat(
        [icbhi_df, asthma_df],
        ignore_index=True
    )

    return result


def validate_classes(df):
    print("\nValidating project classes...")

    actual_classes = sorted(
        df["class_name"].unique().tolist()
    )

    expected_classes = sorted(CLASSES)

    print(
        f"Expected classes: {expected_classes}"
    )

    print(
        f"Actual classes:   {actual_classes}"
    )

    missing = set(expected_classes) - set(actual_classes)
    extra = set(actual_classes) - set(expected_classes)

    if missing:
        raise ValueError(
            f"Missing project classes: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"Unexpected project classes: {sorted(extra)}"
        )

    if "Bronchial" in actual_classes:
        raise ValueError(
            "Project class 'Bronchial' is still present. "
            "It must be mapped to 'Bronchitis'."
        )

    print("Class validation passed.")


def split_patient_groups(df):
    print("\nCreating patient level 70/15/15 split...")

    rng = random.Random(SEED)

    split_assignments = {}

    for class_name in CLASSES:
        class_df = df[
            df["class_name"] == class_name
        ]

        groups = sorted(
            class_df[
                "patient_group"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        rng.shuffle(groups)

        n_groups = len(groups)

        if n_groups < 3:
            raise ValueError(
                f"Class {class_name} has only "
                f"{n_groups} patient groups."
            )

        n_test = max(
            1,
            round(n_groups * TEST_RATIO)
        )

        n_val = max(
            1,
            round(n_groups * VAL_RATIO)
        )

        if n_test + n_val >= n_groups:
            n_test = 1
            n_val = 1

        test_groups = groups[:n_test]

        val_groups = groups[
            n_test:n_test + n_val
        ]

        train_groups = groups[
            n_test + n_val:
        ]

        for group in train_groups:
            split_assignments[group] = "train"

        for group in val_groups:
            split_assignments[group] = "validation"

        for group in test_groups:
            split_assignments[group] = "test"

        print(
            f"{class_name}: "
            f"{len(train_groups)} train groups, "
            f"{len(val_groups)} validation groups, "
            f"{len(test_groups)} test groups"
        )

    df = df.copy()

    df["split"] = df[
        "patient_group"
    ].map(split_assignments)

    if df["split"].isna().any():
        missing_groups = (
            df.loc[
                df["split"].isna(),
                "patient_group"
            ]
            .unique()
        )

        raise ValueError(
            f"Missing split assignment for groups: "
            f"{missing_groups}"
        )

    return df


def verify_patient_disjointness(df):
    print("\nChecking patient group separation...")

    train_groups = set(
        df.loc[
            df["split"] == "train",
            "patient_group"
        ]
    )

    val_groups = set(
        df.loc[
            df["split"] == "validation",
            "patient_group"
        ]
    )

    test_groups = set(
        df.loc[
            df["split"] == "test",
            "patient_group"
        ]
    )

    train_val = train_groups & val_groups
    train_test = train_groups & test_groups
    val_test = val_groups & test_groups

    print(
        f"Train groups: {len(train_groups)}"
    )

    print(
        f"Validation groups: {len(val_groups)}"
    )

    print(
        f"Test groups: {len(test_groups)}"
    )

    if train_val:
        raise ValueError(
            "Patient leakage between train and "
            f"validation: {train_val}"
        )

    if train_test:
        raise ValueError(
            "Patient leakage between train and "
            f"test: {train_test}"
        )

    if val_test:
        raise ValueError(
            "Patient leakage between validation and "
            f"test: {val_test}"
        )

    print("Patient level separation passed.")


def verify_class_presence(df):
    print("\nChecking class presence in every split...")

    for split_name in [
        "train",
        "validation",
        "test"
    ]:
        split_df = df[
            df["split"] == split_name
        ]

        classes = set(
            split_df["class_name"].unique()
        )

        missing = set(CLASSES) - classes

        print(
            f"{split_name}: "
            f"{len(split_df)} samples, "
            f"{len(classes)} classes"
        )

        if missing:
            raise ValueError(
                f"{split_name} is missing classes: "
                f"{sorted(missing)}"
            )

    print(
        "All five classes are present in every split."
    )


def create_reports(df):
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    class_counts = (
        df.groupby(
            ["split", "class_name"]
        )
        .size()
        .reset_index(name="samples")
    )

    class_counts["percentage"] = (
        class_counts.groupby("split")["samples"]
        .transform(
            lambda x: 100 * x / x.sum()
        )
    )

    class_counts.to_csv(
        REPORT_DIR / "class_distribution.csv",
        index=False
    )

    source_counts = (
        df.groupby(
            ["split", "source"]
        )
        .size()
        .reset_index(name="samples")
    )

    source_counts.to_csv(
        REPORT_DIR / "source_distribution.csv",
        index=False
    )

    source_class_counts = (
        df.groupby(
            [
                "split",
                "source",
                "source_class",
                "class_name"
            ]
        )
        .size()
        .reset_index(name="samples")
    )

    source_class_counts.to_csv(
        REPORT_DIR / "source_class_distribution.csv",
        index=False
    )

    patient_counts = (
        df.groupby(
            ["split", "class_name"]
        )["patient_group"]
        .nunique()
        .reset_index(
            name="patient_groups"
        )
    )

    patient_counts.to_csv(
        REPORT_DIR / "patient_group_distribution.csv",
        index=False
    )


def save_outputs(df):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = df.copy()

    df = df.sort_values(
        [
            "split",
            "class_name",
            "source",
            "patient_group",
            "sample_id"
        ]
    ).reset_index(drop=True)

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "validation"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    df.to_csv(
        UNIFIED_METADATA_FILE,
        index=False
    )

    train_df.to_csv(
        TRAIN_METADATA_FILE,
        index=False
    )

    val_df.to_csv(
        VAL_METADATA_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_METADATA_FILE,
        index=False
    )

    print("\nSaved:")
    print(UNIFIED_METADATA_FILE)
    print(TRAIN_METADATA_FILE)
    print(VAL_METADATA_FILE)
    print(TEST_METADATA_FILE)


def print_final_summary(df):
    print("\n" + "=" * 70)
    print("FINAL UNIFIED DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Total samples: {len(df)}"
    )

    print("\nOverall project class counts:")

    print(
        df["class_name"]
        .value_counts()
        .reindex(CLASSES)
        .fillna(0)
        .astype(int)
    )

    print("\nSplit counts:")

    print(
        df.groupby(
            ["split", "class_name"]
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=CLASSES,
            fill_value=0
        )
    )

    print("\nSource counts:")

    print(
        df["source"].value_counts()
    )

    print("\nSource and project class mapping:")

    print(
        df[
            [
                "source",
                "source_class",
                "class_name"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "source",
                "source_class"
            ]
        )
        .to_string(index=False)
    )

    print("\nPatient groups by split:")

    print(
        df.groupby("split")[
            "patient_group"
        ]
        .nunique()
    )

    print(
        "\nBronchial source label mapping check:"
    )

    bronchial_rows = df[
        df["source_class"]
        .astype(str)
        .str.lower()
        == "bronchial"
    ]

    if len(bronchial_rows) == 0:
        print(
            "No Asthma V2 Bronchial samples found."
        )
    else:
        mapped_classes = (
            bronchial_rows[
                "class_name"
            ]
            .unique()
            .tolist()
        )

        print(
            "Source class: Bronchial -> "
            f"Project class: {mapped_classes}"
        )

        if mapped_classes != ["Bronchitis"]:
            raise ValueError(
                "Bronchial samples were not mapped "
                "exclusively to Bronchitis."
            )

    print("=" * 70)


def main():
    set_seed()

    print("=" * 70)
    print(
        "CREATING UNIFIED RESPIRATORY AUDIO DATASET"
    )
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"ICBHI directory: {ICBHI_DIR}"
    )

    print(
        f"Asthma V2 directory: {ASTHMA_V2_DIR}"
    )

    if not ICBHI_DIR.exists():
        raise FileNotFoundError(
            f"ICBHI directory not found: {ICBHI_DIR}"
        )

    if not DIAGNOSIS_FILE.exists():
        raise FileNotFoundError(
            f"Diagnosis CSV not found: {DIAGNOSIS_FILE}"
        )

    if not ASTHMA_V2_DIR.exists():
        raise FileNotFoundError(
            f"Asthma V2 directory not found: {ASTHMA_V2_DIR}"
        )

    icbhi_df = collect_icbhi_cycles()

    asthma_v2_df = collect_asthma_v2_recordings()

    if icbhi_df.empty:
        raise ValueError(
            "No ICBHI samples were collected."
        )

    if asthma_v2_df.empty:
        raise ValueError(
            "No Asthma V2 recordings were collected."
        )

    print("\nCombining datasets...")

    df = pd.concat(
        [
            icbhi_df,
            asthma_v2_df
        ],
        ignore_index=True
    )

    print(
        f"Samples before duplicate removal: {len(df)}"
    )

    df = remove_asthma_v2_duplicates(df)

    print(
        f"Samples after duplicate removal: {len(df)}"
    )

    validate_classes(df)

    df = split_patient_groups(df)

    verify_patient_disjointness(df)

    verify_class_presence(df)

    create_reports(df)

    save_outputs(df)

    print_final_summary(df)

    print(
        "\nDataset creation completed successfully."
    )


if __name__ == "__main__":
    main()