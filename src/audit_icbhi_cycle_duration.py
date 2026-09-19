import os
import glob
import re
import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"
ICBHI_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "audio_and_text_files"
)
OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "audit"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def get_patient_id(filename):
    match = re.match(r"(\d+)_", filename)
    if match:
        return match.group(1)
    return ""


def read_annotation_file(path):
    rows = []

    with open(path, "r", encoding="utf8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                start_time = float(parts[0])
                end_time = float(parts[1])
            except ValueError:
                continue

            duration = end_time - start_time

            crackles = ""
            wheezes = ""

            if len(parts) >= 3:
                crackles = parts[2]

            if len(parts) >= 4:
                wheezes = parts[3]

            rows.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "crackles": crackles,
                "wheezes": wheezes,
                "line_number": line_number
            })

    return rows


def audit_cycle_durations():
    annotation_files = sorted(
        glob.glob(
            os.path.join(
                ICBHI_ROOT,
                "*.txt"
            )
        )
    )

    print("=" * 70)
    print("ICBHI RESPIRATORY CYCLE DURATION AUDIT")
    print("=" * 70)

    print()
    print(f"ICBHI root:")
    print(ICBHI_ROOT)

    print()
    print(f"Annotation files found: {len(annotation_files)}")

    records = []

    for annotation_path in annotation_files:
        filename = os.path.basename(annotation_path)
        patient_id = get_patient_id(filename)

        cycles = read_annotation_file(annotation_path)

        for cycle_index, cycle in enumerate(cycles, start=1):
            record = {
                "patient_id": patient_id,
                "filename": filename,
                "cycle_index": cycle_index,
                "start_time": cycle["start_time"],
                "end_time": cycle["end_time"],
                "duration": cycle["duration"],
                "crackles": cycle["crackles"],
                "wheezes": cycle["wheezes"]
            }

            records.append(record)

    df = pd.DataFrame(records)

    print()
    print(f"Total annotated cycles: {len(df)}")

    invalid = df[
        (df["duration"] <= 0)
        | (~np.isfinite(df["duration"]))
    ]

    print()
    print(f"Invalid cycle durations: {len(invalid)}")

    df = df[
        (df["duration"] > 0)
        & np.isfinite(df["duration"])
    ].copy()

    durations = df["duration"].to_numpy()

    print()
    print("=" * 70)
    print("DURATION STATISTICS")
    print("=" * 70)

    print()
    print(f"Minimum: {np.min(durations):.4f} seconds")
    print(f"Maximum: {np.max(durations):.4f} seconds")
    print(f"Mean:    {np.mean(durations):.4f} seconds")
    print(f"Median:  {np.median(durations):.4f} seconds")

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    print()
    print("Percentiles:")

    for percentile in percentiles:
        value = np.percentile(
            durations,
            percentile
        )

        print(
            f"  {percentile:2d}th percentile: "
            f"{value:.4f} seconds"
        )

    print()
    print("=" * 70)
    print("CYCLES ABOVE DURATION THRESHOLDS")
    print("=" * 70)

    thresholds = [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        6.0,
        8.0,
        10.0
    ]

    threshold_rows = []

    for threshold in thresholds:
        count = int(
            np.sum(durations >= threshold)
        )

        percentage = (
            count / len(durations) * 100
        )

        threshold_rows.append({
            "threshold_seconds": threshold,
            "cycles": count,
            "percentage": percentage
        })

        print(
            f"At least {threshold:4.1f} seconds: "
            f"{count:4d} cycles "
            f"({percentage:6.2f}%)"
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    print()
    print("=" * 70)
    print("CYCLE DURATION BY SOUND CLASS")
    print("=" * 70)

    df["sound_class"] = np.select(
        [
            (df["crackles"] == "1")
            & (df["wheezes"] == "1"),
            (df["crackles"] == "1")
            & (df["wheezes"] == "0"),
            (df["crackles"] == "0")
            & (df["wheezes"] == "1"),
            (df["crackles"] == "0")
            & (df["wheezes"] == "0")
        ],
        [
            "Both",
            "Crackles",
            "Wheezes",
            "Normal"
        ],
        default="Unknown"
    )

    class_summary = (
        df.groupby("sound_class")["duration"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            minimum="min",
            maximum="max"
        )
        .reset_index()
    )

    print()
    print(
        class_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    cycle_output = os.path.join(
        OUTPUT_ROOT,
        "icbhi_cycle_durations.csv"
    )

    threshold_output = os.path.join(
        OUTPUT_ROOT,
        "icbhi_cycle_duration_thresholds.csv"
    )

    class_output = os.path.join(
        OUTPUT_ROOT,
        "icbhi_cycle_duration_by_sound_class.csv"
    )

    df.to_csv(
        cycle_output,
        index=False
    )

    threshold_df.to_csv(
        threshold_output,
        index=False
    )

    class_summary.to_csv(
        class_output,
        index=False
    )

    print()
    print("=" * 70)
    print("REPORTS SAVED")
    print("=" * 70)

    print()
    print(cycle_output)
    print(threshold_output)
    print(class_output)

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    audit_cycle_durations()