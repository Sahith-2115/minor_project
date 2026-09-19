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

DIAGNOSIS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "patient_diagnosis.csv"
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
    "Pneumonia"
]

CAPS = [
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50
]


def get_patient_id(filename):
    match = re.match(r"(\d+)_", filename)

    if match:
        return int(match.group(1))

    return None


def load_diagnoses():
    df = pd.read_csv(DIAGNOSIS_FILE)

    diagnosis_map = {}

    for _, row in df.iterrows():
        patient_id = int(row["ID"])
        diagnosis = str(row["Symptom"]).strip()
        diagnosis_map[patient_id] = diagnosis

    return diagnosis_map


def count_cycles_per_patient():
    diagnosis_map = load_diagnoses()

    patient_counts = {}

    annotation_files = [
        f for f in os.listdir(ICBHI_ROOT)
        if f.lower().endswith(".txt")
    ]

    for filename in annotation_files:
        patient_id = get_patient_id(filename)

        if patient_id is None:
            continue

        diagnosis = diagnosis_map.get(patient_id)

        if diagnosis not in TARGET_CLASSES:
            continue

        path = os.path.join(
            ICBHI_ROOT,
            filename
        )

        cycle_count = 0

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 4:
                    continue

                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    int(parts[2])
                    int(parts[3])
                except ValueError:
                    continue

                if end > start:
                    cycle_count += 1

        if patient_id not in patient_counts:
            patient_counts[patient_id] = {
                "patient_id": patient_id,
                "class_name": diagnosis,
                "cycles": 0,
                "annotation_files": 0
            }

        patient_counts[patient_id]["cycles"] += cycle_count
        patient_counts[patient_id]["annotation_files"] += 1

    return pd.DataFrame(
        list(patient_counts.values())
    )


def calculate_capped_counts(df):
    rows = []

    for cap in CAPS:
        for class_name in TARGET_CLASSES:
            class_df = df[
                df["class_name"] == class_name
            ]

            raw_total = int(
                class_df["cycles"].sum()
            )

            capped_total = int(
                class_df["cycles"]
                .clip(upper=cap)
                .sum()
            )

            patients = len(class_df)

            rows.append({
                "cap_per_patient": cap,
                "class_name": class_name,
                "patients": patients,
                "raw_cycles": raw_total,
                "capped_cycles": capped_total,
                "cycles_removed": raw_total - capped_total,
                "retained_percent": round(
                    100 * capped_total / raw_total,
                    2
                )
            })

    return pd.DataFrame(rows)


def print_patient_counts(df):
    print()
    print("==============================================")
    print("ICBHI TRUE PATIENT LEVEL CYCLE COUNTS")
    print("==============================================")

    for class_name in TARGET_CLASSES:
        print()
        print(class_name)

        class_df = df[
            df["class_name"] == class_name
        ].sort_values(
            "cycles",
            ascending=False
        )

        print(
            class_df.to_string(index=False)
        )


def print_capped_summary(summary):
    print()
    print("==============================================")
    print("PATIENT CAPPED SAMPLING")
    print("==============================================")

    for cap in CAPS:
        subset = summary[
            summary["cap_per_patient"] == cap
        ]

        print()
        print(
            f"Maximum {cap} cycles per patient"
        )

        for _, row in subset.iterrows():
            print(
                f"{row['class_name']:10s}: "
                f"{row['capped_cycles']:4d} cycles "
                f"from {row['patients']:2d} patients "
                f"({row['retained_percent']:6.2f}% retained)"
            )


def print_annotation_file_summary(df):
    print()
    print("==============================================")
    print("ANNOTATION FILE SUMMARY")
    print("==============================================")

    summary = (
        df.groupby("class_name")
        .agg(
            patients=("patient_id", "count"),
            annotation_files=("annotation_files", "sum"),
            cycles=("cycles", "sum")
        )
        .reset_index()
    )

    print(
        summary.to_string(index=False)
    )


def main():
    print("==============================================")
    print("ICBHI PATIENT CAPPED SAMPLING AUDIT")
    print("==============================================")

    print()
    print("ICBHI root:")
    print(ICBHI_ROOT)

    df = count_cycles_per_patient()

    print_patient_counts(df)
    print_annotation_file_summary(df)

    summary = calculate_capped_counts(df)

    print_capped_summary(summary)

    patient_path = os.path.join(
        OUTPUT_DIR,
        "icbhi_true_patient_cycle_counts.csv"
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "icbhi_true_patient_capped_sampling.csv"
    )

    df.to_csv(
        patient_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print("==============================================")
    print("REPORTS SAVED")
    print("==============================================")

    print()
    print(patient_path)

    print(summary_path)

    print()
    print("==============================================")
    print("AUDIT COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()