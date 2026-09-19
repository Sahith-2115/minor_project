from pathlib import Path
import hashlib
import pandas as pd

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
DATASET_DIR = PROJECT_ROOT / "data/processed/unified_dataset"
REPORT_DIR = DATASET_DIR / "reports"

FILES = {
    "unified": DATASET_DIR / "unified_dataset_metadata.csv",
    "train": DATASET_DIR / "train_metadata.csv",
    "validation": DATASET_DIR / "validation_metadata.csv",
    "test": DATASET_DIR / "test_metadata.csv"
}

CLASSES = [
    "Asthma",
    "Bronchitis",
    "COPD",
    "Healthy",
    "Pneumonia"
]


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_metadata():
    data = {}

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {path}"
            )

        df = pd.read_csv(path)

        data[name] = df

        print(
            f"{name}: {len(df)} rows"
        )

    return data


def check_required_columns(data):
    print("\n" + "=" * 70)
    print("CHECKING REQUIRED COLUMNS")
    print("=" * 70)

    required = {
        "source",
        "source_class",
        "class_name",
        "patient_id",
        "patient_group",
        "audio_path",
        "sample_id",
        "split"
    }

    for name, df in data.items():
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: "
                f"{sorted(missing)}"
            )

        print(
            f"{name}: all required columns present"
        )


def check_total_counts(data):
    print("\n" + "=" * 70)
    print("CHECKING TOTAL COUNTS")
    print("=" * 70)

    unified = data["unified"]

    expected_total = 7552

    print(
        f"Expected total: {expected_total}"
    )

    print(
        f"Actual total:   {len(unified)}"
    )

    if len(unified) != expected_total:
        raise ValueError(
            f"Expected {expected_total} samples, "
            f"found {len(unified)}."
        )

    train_count = len(data["train"])
    val_count = len(data["validation"])
    test_count = len(data["test"])

    split_total = (
        train_count +
        val_count +
        test_count
    )

    print(
        f"Train:       {train_count}"
    )

    print(
        f"Validation:  {val_count}"
    )

    print(
        f"Test:        {test_count}"
    )

    print(
        f"Split total: {split_total}"
    )

    if split_total != expected_total:
        raise ValueError(
            "Train, validation and test totals "
            "do not equal the unified dataset."
        )

    print("Total count check passed.")


def check_class_distribution(data):
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    unified = data["unified"]

    counts = (
        unified["class_name"]
        .value_counts()
        .reindex(CLASSES)
        .fillna(0)
        .astype(int)
    )

    print("\nOverall:")
    print(counts)

    expected = {
        "Asthma": 288,
        "Bronchitis": 104,
        "COPD": 6147,
        "Healthy": 449,
        "Pneumonia": 564
    }

    for class_name in CLASSES:
        actual = int(counts[class_name])

        if actual != expected[class_name]:
            raise ValueError(
                f"{class_name}: expected "
                f"{expected[class_name]}, "
                f"found {actual}"
            )

    print("\nSplit distribution:")

    split_counts = (
        unified.groupby(
            ["split", "class_name"]
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=CLASSES,
            fill_value=0
        )
    )

    print(split_counts)

    print("\nClass percentages:")

    split_percentages = (
        split_counts.div(
            split_counts.sum(axis=1),
            axis=0
        ) * 100
    )

    print(
        split_percentages.round(2)
    )

    print("\nClass distribution check passed.")


def check_split_consistency(data):
    print("\n" + "=" * 70)
    print("CHECKING SPLIT CONSISTENCY")
    print("=" * 70)

    unified = data["unified"]

    expected_splits = {
        "train": data["train"],
        "validation": data["validation"],
        "test": data["test"]
    }

    for split_name, split_df in expected_splits.items():
        actual = unified[
            unified["split"] == split_name
        ]

        unified_ids = set(
            actual["sample_id"]
        )

        split_ids = set(
            split_df["sample_id"]
        )

        missing = split_ids - unified_ids
        extra = unified_ids - split_ids

        print(
            f"{split_name}: "
            f"{len(split_df)} metadata rows"
        )

        if missing:
            raise ValueError(
                f"{split_name}: samples missing "
                f"from unified metadata: {missing}"
            )

        if extra:
            raise ValueError(
                f"{split_name}: extra samples "
                f"found in unified metadata: {extra}"
            )

        if len(unified_ids) != len(split_ids):
            raise ValueError(
                f"{split_name}: duplicate sample IDs "
                f"detected."
            )

    print("Split consistency passed.")


def check_sample_ids(data):
    print("\n" + "=" * 70)
    print("CHECKING SAMPLE IDs")
    print("=" * 70)

    unified = data["unified"]

    duplicate_ids = unified[
        unified["sample_id"].duplicated(
            keep=False
        )
    ]

    if not duplicate_ids.empty:
        print(
            duplicate_ids[
                ["sample_id", "source", "audio_path"]
            ].to_string(index=False)
        )

        raise ValueError(
            "Duplicate sample IDs detected."
        )

    print(
        f"Unique sample IDs: "
        f"{unified['sample_id'].nunique()}"
    )

    print("Sample ID check passed.")


def check_patient_disjointness(data):
    print("\n" + "=" * 70)
    print("CHECKING PATIENT LEVEL SEPARATION")
    print("=" * 70)

    unified = data["unified"]

    train_groups = set(
        unified.loc[
            unified["split"] == "train",
            "patient_group"
        ]
    )

    val_groups = set(
        unified.loc[
            unified["split"] == "validation",
            "patient_group"
        ]
    )

    test_groups = set(
        unified.loc[
            unified["split"] == "test",
            "patient_group"
        ]
    )

    train_val = train_groups & val_groups
    train_test = train_groups & test_groups
    val_test = val_groups & test_groups

    print(
        f"Train groups:       {len(train_groups)}"
    )

    print(
        f"Validation groups:  {len(val_groups)}"
    )

    print(
        f"Test groups:        {len(test_groups)}"
    )

    if train_val:
        raise ValueError(
            f"Train/validation patient overlap: "
            f"{sorted(train_val)}"
        )

    if train_test:
        raise ValueError(
            f"Train/test patient overlap: "
            f"{sorted(train_test)}"
        )

    if val_test:
        raise ValueError(
            f"Validation/test patient overlap: "
            f"{sorted(val_test)}"
        )

    print(
        "No patient group overlap detected."
    )


def check_source_distribution(data):
    print("\n" + "=" * 70)
    print("SOURCE DISTRIBUTION")
    print("=" * 70)

    unified = data["unified"]

    print("\nOverall source counts:")
    print(
        unified["source"]
        .value_counts()
    )

    print("\nSource by split:")

    source_split = (
        unified.groupby(
            ["split", "source"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(source_split)

    print("\nSource by split and class:")

    source_class_split = (
        unified.groupby(
            [
                "split",
                "source",
                "class_name"
            ]
        )
        .size()
        .reset_index(
            name="samples"
        )
    )

    print(
        source_class_split.to_string(
            index=False
        )
    )

    source_class_split.to_csv(
        REPORT_DIR / "final_source_class_audit.csv",
        index=False
    )


def check_class_mapping(data):
    print("\n" + "=" * 70)
    print("CHECKING SOURCE TO PROJECT CLASS MAPPING")
    print("=" * 70)

    unified = data["unified"]

    mapping = (
        unified[
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
    )

    print(
        mapping.to_string(index=False)
    )

    bronchial = unified[
        (
            unified["source"] == "Asthma_V2"
        )
        &
        (
            unified["source_class"]
            .astype(str)
            .str.lower()
            == "bronchial"
        )
    ]

    if not bronchial.empty:
        project_classes = (
            bronchial["class_name"]
            .unique()
            .tolist()
        )

        if project_classes != ["Bronchitis"]:
            raise ValueError(
                "Asthma V2 Bronchial samples "
                "are not mapped exclusively "
                "to Bronchitis."
            )

    print(
        "Bronchial to Bronchitis mapping passed."
    )


def check_audio_paths(data):
    print("\n" + "=" * 70)
    print("CHECKING AUDIO PATHS")
    print("=" * 70)

    unified = data["unified"]

    missing = []

    for path in unified["audio_path"].drop_duplicates():
        if not Path(path).exists():
            missing.append(path)

    print(
        f"Unique audio paths: "
        f"{unified['audio_path'].nunique()}"
    )

    print(
        f"Missing audio paths: "
        f"{len(missing)}"
    )

    if missing:
        print("\nFirst missing paths:")

        for path in missing[:20]:
            print(path)

        raise FileNotFoundError(
            f"{len(missing)} audio paths do not exist."
        )

    print("All referenced audio files exist.")


def check_cross_split_audio_paths(data):
    print("\n" + "=" * 70)
    print("CHECKING CROSS SPLIT AUDIO PATH OVERLAP")
    print("=" * 70)

    unified = data["unified"]

    split_paths = {}

    for split_name in [
        "train",
        "validation",
        "test"
    ]:
        split_paths[split_name] = set(
            unified.loc[
                unified["split"] == split_name,
                "audio_path"
            ]
        )

    train_val = (
        split_paths["train"]
        & split_paths["validation"]
    )

    train_test = (
        split_paths["train"]
        & split_paths["test"]
    )

    val_test = (
        split_paths["validation"]
        & split_paths["test"]
    )

    print(
        f"Train/validation shared audio paths: "
        f"{len(train_val)}"
    )

    print(
        f"Train/test shared audio paths: "
        f"{len(train_test)}"
    )

    print(
        f"Validation/test shared audio paths: "
        f"{len(val_test)}"
    )

    if train_val:
        raise ValueError(
            "Same audio recording appears in "
            "train and validation."
        )

    if train_test:
        raise ValueError(
            "Same audio recording appears in "
            "train and test."
        )

    if val_test:
        raise ValueError(
            "Same audio recording appears in "
            "validation and test."
        )

    print(
        "No cross split audio path overlap."
    )


def check_exact_audio_hash_overlap(data):
    print("\n" + "=" * 70)
    print("CHECKING CROSS SPLIT EXACT AUDIO HASH OVERLAP")
    print("=" * 70)

    unified = data["unified"]

    unique_audio = (
        unified[
            [
                "audio_path",
                "split"
            ]
        ]
        .drop_duplicates()
    )

    hash_to_paths = {}

    for _, row in unique_audio.iterrows():
        path = Path(row["audio_path"])

        file_hash = sha256_file(path)

        if file_hash not in hash_to_paths:
            hash_to_paths[file_hash] = []

        hash_to_paths[file_hash].append(
            (
                row["split"],
                str(path)
            )
        )

    cross_split_groups = []

    for file_hash, entries in hash_to_paths.items():
        splits = set(
            entry[0]
            for entry in entries
        )

        if len(splits) > 1:
            cross_split_groups.append(
                (
                    file_hash,
                    entries
                )
            )

    print(
        f"Unique audio files hashed: "
        f"{len(unique_audio)}"
    )

    print(
        f"Cross split exact hash groups: "
        f"{len(cross_split_groups)}"
    )

    if cross_split_groups:
        print(
            "\nFirst cross split hash groups:"
        )

        for file_hash, entries in (
            cross_split_groups[:10]
        ):
            print(
                f"\nHash: {file_hash}"
            )

            for split_name, path in entries:
                print(
                    f"  {split_name}: {path}"
                )

        raise ValueError(
            "Exact audio content appears "
            "across different splits."
        )

    print(
        "No cross split exact audio hash overlap."
    )


def check_duplicate_asthma_v2(data):
    print("\n" + "=" * 70)
    print("CHECKING ASTHMA V2 EXACT DUPLICATES")
    print("=" * 70)

    unified = data["unified"]

    asthma_v2 = unified[
        unified["source"] == "Asthma_V2"
    ].copy()

    hashes = {}

    for path in asthma_v2[
        "audio_path"
    ].drop_duplicates():

        file_hash = sha256_file(
            Path(path)
        )

        hashes.setdefault(
            file_hash,
            []
        ).append(path)

    duplicate_groups = {
        h: paths
        for h, paths in hashes.items()
        if len(paths) > 1
    }

    print(
        f"Asthma V2 unique recordings: "
        f"{len(asthma_v2)}"
    )

    print(
        f"Asthma V2 duplicate hash groups "
        f"remaining: {len(duplicate_groups)}"
    )

    if duplicate_groups:
        for file_hash, paths in list(
            duplicate_groups.items()
        )[:10]:

            print(
                f"\nHash: {file_hash}"
            )

            for path in paths:
                print(path)

        raise ValueError(
            "Duplicate Asthma V2 recordings "
            "remain after preprocessing."
        )

    print(
        "Asthma V2 duplicate check passed."
    )


def check_icbhi_cycle_structure(data):
    print("\n" + "=" * 70)
    print("CHECKING ICBHI CYCLE STRUCTURE")
    print("=" * 70)

    unified = data["unified"]

    icbhi = unified[
        unified["source"] == "ICBHI"
    ]

    print(
        f"ICBHI cycles: {len(icbhi)}"
    )

    print(
        f"ICBHI patient groups: "
        f"{icbhi['patient_group'].nunique()}"
    )

    print("\nICBHI class counts:")

    print(
        icbhi["class_name"]
        .value_counts()
        .sort_index()
    )

    invalid_duration = icbhi[
        (
            icbhi["duration"] <= 0
        )
        |
        (
            icbhi["end_time"]
            <= icbhi["start_time"]
        )
    ]

    print(
        f"\nInvalid cycle durations: "
        f"{len(invalid_duration)}"
    )

    if not invalid_duration.empty:
        raise ValueError(
            "Invalid ICBHI cycle durations detected."
        )

    print(
        "ICBHI cycle structure passed."
    )


def check_patient_group_counts(data):
    print("\n" + "=" * 70)
    print("PATIENT GROUP DISTRIBUTION")
    print("=" * 70)

    unified = data["unified"]

    patient_counts = (
        unified.groupby(
            [
                "split",
                "class_name"
            ]
        )["patient_group"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(
            columns=CLASSES,
            fill_value=0
        )
    )

    print(patient_counts)

    patient_counts.to_csv(
        REPORT_DIR / "final_patient_group_audit.csv"
    )


def save_final_audit(data):
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    unified = data["unified"]

    summary = []

    for split_name in [
        "train",
        "validation",
        "test"
    ]:
        split_df = unified[
            unified["split"] == split_name
        ]

        for class_name in CLASSES:
            class_df = split_df[
                split_df["class_name"] == class_name
            ]

            summary.append({
                "split": split_name,
                "class_name": class_name,
                "samples": len(class_df),
                "patient_groups": class_df[
                    "patient_group"
                ].nunique(),
                "icbhi_samples": int(
                    (
                        class_df["source"]
                        == "ICBHI"
                    ).sum()
                ),
                "asthma_v2_samples": int(
                    (
                        class_df["source"]
                        == "Asthma_V2"
                    ).sum()
                )
            })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        REPORT_DIR / "final_dataset_audit_summary.csv",
        index=False
    )

    print(
        f"\nAudit summary saved to:\n"
        f"{REPORT_DIR / 'final_dataset_audit_summary.csv'}"
    )


def main():
    print("=" * 70)
    print("FINAL UNIFIED DATASET AUDIT")
    print("=" * 70)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data = load_metadata()

    check_required_columns(data)
    check_total_counts(data)
    check_class_distribution(data)
    check_split_consistency(data)
    check_sample_ids(data)
    check_patient_disjointness(data)
    check_source_distribution(data)
    check_class_mapping(data)
    check_audio_paths(data)
    check_cross_split_audio_paths(data)
    check_exact_audio_hash_overlap(data)
    check_duplicate_asthma_v2(data)
    check_icbhi_cycle_structure(data)
    check_patient_group_counts(data)
    save_final_audit(data)

    print("\n" + "=" * 70)
    print("FINAL DATASET AUDIT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()