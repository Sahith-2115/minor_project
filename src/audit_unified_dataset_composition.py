import os
import re
import pandas as pd

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

ICBHI_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "audio_and_text_files"
)

ICBHI_DIAGNOSIS = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "patient_diagnosis.csv"
)

ASTHMA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "asthma_detection_v2",
    "Asthma Detection Dataset Version 2"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLASSES = [
    "Healthy",
    "COPD",
    "Pneumonia",
    "Asthma",
    "Bronchial"
]


def normalize_icbhi_diagnosis(value):
    value = str(value).strip()

    mapping = {
        "COPD": "COPD",
        "Healthy": "Healthy",
        "Pneumonia": "Pneumonia",
        "Asthma": "Asthma",
        "Bronchiectasis": "Bronchiectasis",
        "Bronchiolitis": "Bronchiolitis",
        "URTI": "URTI",
        "LRTI": "LRTI"
    }

    return mapping.get(value, value)


def get_icbhi_patient_id(filename):
    match = re.match(r"(\d+)_", filename)

    if match:
        return int(match.group(1))

    return None


def get_cycle_class(crackles, wheezes):
    crackles = int(crackles)
    wheezes = int(wheezes)

    if crackles == 1 and wheezes == 1:
        return "Both"

    if crackles == 1:
        return "Crackles"

    if wheezes == 1:
        return "Wheezes"

    return "Normal"


def audit_icbhi():
    diagnosis_df = pd.read_csv(ICBHI_DIAGNOSIS)

    diagnosis_map = {}

    for _, row in diagnosis_df.iterrows():
        patient_id = int(row["ID"])
        diagnosis = normalize_icbhi_diagnosis(row["Symptom"])
        diagnosis_map[patient_id] = diagnosis

    records = []

    annotation_files = [
        f for f in os.listdir(ICBHI_ROOT)
        if f.lower().endswith(".txt")
    ]

    for annotation_file in annotation_files:
        patient_id = get_icbhi_patient_id(annotation_file)

        if patient_id is None:
            continue

        diagnosis = diagnosis_map.get(patient_id)

        if diagnosis not in TARGET_CLASSES:
            continue

        annotation_path = os.path.join(
            ICBHI_ROOT,
            annotation_file
        )

        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 4:
                    continue

                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    crackles = int(parts[2])
                    wheezes = int(parts[3])
                except ValueError:
                    continue

                duration = end - start

                if duration <= 0:
                    continue

                sound_class = get_cycle_class(
                    crackles,
                    wheezes
                )

                records.append({
                    "source": "ICBHI",
                    "patient_group": f"ICBHI_{patient_id}",
                    "patient_id": patient_id,
                    "class_name": diagnosis,
                    "sample_type": "cycle",
                    "duration": duration,
                    "sound_class": sound_class
                })

    return pd.DataFrame(records)


def get_asthma_patient_id(filename):
    match = re.match(r"(P\d+)", filename)

    if match:
        return match.group(1)

    return "UNKNOWN"


def audit_asthma_v2():
    records = []

    class_folders = [
        d for d in os.listdir(ASTHMA_ROOT)
        if os.path.isdir(os.path.join(ASTHMA_ROOT, d))
    ]

    class_mapping = {
        "healthy": "Healthy",
        "copd": "COPD",
        "pneumonia": "Pneumonia",
        "asthma": "Asthma",
        "bronchial": "Bronchial"
    }

    for folder in class_folders:
        class_name = class_mapping.get(folder.lower())

        if class_name not in TARGET_CLASSES:
            continue

        folder_path = os.path.join(
            ASTHMA_ROOT,
            folder
        )

        wav_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".wav")
        ]

        for wav_file in wav_files:
            patient_id = get_asthma_patient_id(wav_file)

            records.append({
                "source": "Asthma_V2",
                "patient_group": f"Asthma_V2_{class_name}_{patient_id}",
                "patient_id": patient_id,
                "class_name": class_name,
                "sample_type": "recording",
                "duration": None,
                "sound_class": None
            })

    return pd.DataFrame(records)


def print_class_summary(df):
    print()
    print("==============================================")
    print("CLASS SUMMARY")
    print("==============================================")

    summary = (
        df.groupby(
            ["source", "class_name"]
        )
        .agg(
            samples=("class_name", "size"),
            patient_groups=("patient_group", "nunique")
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    summary_path = os.path.join(
        OUTPUT_DIR,
        "unified_class_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print("Saved:")
    print(summary_path)


def print_combined_summary(df):
    print()
    print("==============================================")
    print("COMBINED CLASS SUMMARY")
    print("==============================================")

    summary = (
        df.groupby("class_name")
        .agg(
            samples=("class_name", "size"),
            patient_groups=("patient_group", "nunique")
        )
        .reset_index()
    )

    summary["samples_per_patient_group"] = (
        summary["samples"] /
        summary["patient_groups"]
    ).round(2)

    print(summary.to_string(index=False))

    summary_path = os.path.join(
        OUTPUT_DIR,
        "unified_combined_class_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print("Saved:")
    print(summary_path)


def print_patient_distribution(df):
    print()
    print("==============================================")
    print("PATIENT GROUP DISTRIBUTION")
    print("==============================================")

    patient_summary = (
        df.groupby(
            ["source", "class_name", "patient_group"]
        )
        .size()
        .reset_index(name="samples")
    )

    distribution = (
        patient_summary.groupby(
            ["source", "class_name"]
        )["samples"]
        .agg(
            patients="count",
            minimum="min",
            median="median",
            mean="mean",
            maximum="max"
        )
        .reset_index()
    )

    print(distribution.to_string(index=False))

    path = os.path.join(
        OUTPUT_DIR,
        "unified_patient_group_distribution.csv"
    )

    distribution.to_csv(
        path,
        index=False
    )

    print()
    print("Saved:")
    print(path)


def print_sampling_options(df):
    print()
    print("==============================================")
    print("SAMPLING OPTIONS")
    print("==============================================")

    class_counts = (
        df.groupby("class_name")
        .size()
        .sort_values(ascending=False)
    )

    print()
    print("Available samples by class:")

    for class_name, count in class_counts.items():
        print(
            f"{class_name:12s}: {count:5d}"
        )

    print()
    print("Balanced dataset sizes:")
    print()

    minimum_count = class_counts.min()

    for target in [
        100,
        150,
        200,
        250,
        300,
        400
    ]:
        possible = all(
            count >= target
            for count in class_counts
        )

        if possible:
            total = target * len(class_counts)

            print(
                f"{target:3d} per class -> "
                f"{total:4d} total samples"
            )

    print()
    print("Maximum equal class size:")

    print(
        f"{minimum_count} samples per class"
    )

    print(
        f"{minimum_count * len(class_counts)} "
        f"total samples"
    )


def print_source_contribution(df):
    print()
    print("==============================================")
    print("SOURCE CONTRIBUTION")
    print("==============================================")

    source_summary = (
        df.groupby(
            ["class_name", "source"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    print(source_summary.to_string(index=False))

    path = os.path.join(
        OUTPUT_DIR,
        "unified_source_contribution.csv"
    )

    source_summary.to_csv(
        path,
        index=False
    )

    print()
    print("Saved:")
    print(path)


def print_icbhi_cycle_duration_summary(df):
    icbhi = df[
        df["source"] == "ICBHI"
    ].copy()

    if icbhi.empty:
        return

    print()
    print("==============================================")
    print("ICBHI TARGET CLASS DURATION SUMMARY")
    print("==============================================")

    summary = (
        icbhi.groupby("class_name")["duration"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            minimum="min",
            maximum="max"
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    path = os.path.join(
        OUTPUT_DIR,
        "unified_icbhi_target_duration_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print()
    print("Saved:")
    print(path)


def print_icbhi_patient_counts(df):
    icbhi = df[
        df["source"] == "ICBHI"
    ].copy()

    if icbhi.empty:
        return

    print()
    print("==============================================")
    print("ICBHI PATIENT LEVEL SAMPLE COUNTS")
    print("==============================================")

    patient_summary = (
        icbhi.groupby(
            ["class_name", "patient_id"]
        )
        .size()
        .reset_index(name="cycles")
    )

    for class_name in TARGET_CLASSES:
        class_df = patient_summary[
            patient_summary["class_name"] == class_name
        ]

        if class_df.empty:
            continue

        print()
        print(f"{class_name}:")
        print(
            class_df.to_string(index=False)
        )

    path = os.path.join(
        OUTPUT_DIR,
        "icbhi_patient_cycle_counts.csv"
    )

    patient_summary.to_csv(
        path,
        index=False
    )

    print()
    print("Saved:")
    print(path)


def main():
    print("==============================================")
    print("UNIFIED DATASET COMPOSITION AUDIT")
    print("==============================================")

    print()
    print("ICBHI root:")
    print(ICBHI_ROOT)

    print()
    print("Asthma V2 root:")
    print(ASTHMA_ROOT)

    icbhi_df = audit_icbhi()
    asthma_df = audit_asthma_v2()

    print()
    print("ICBHI target class samples:")
    print(len(icbhi_df))

    print()
    print("Asthma V2 target class samples:")
    print(len(asthma_df))

    unified_df = pd.concat(
        [icbhi_df, asthma_df],
        ignore_index=True
    )

    print()
    print("Total unified samples:")
    print(len(unified_df))

    print_class_summary(unified_df)
    print_combined_summary(unified_df)
    print_patient_distribution(unified_df)
    print_source_contribution(unified_df)
    print_icbhi_cycle_duration_summary(unified_df)
    print_icbhi_patient_counts(unified_df)
    print_sampling_options(unified_df)

    full_path = os.path.join(
        OUTPUT_DIR,
        "unified_dataset_composition.csv"
    )

    unified_df.to_csv(
        full_path,
        index=False
    )

    print()
    print("==============================================")
    print("AUDIT COMPLETE")
    print("==============================================")

    print()
    print("Full audit file:")
    print(full_path)


if __name__ == "__main__":
    main()