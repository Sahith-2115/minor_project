import os
import sys
import pandas as pd

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

SPLIT_FILES = {
    "validation": os.path.join(
        PROJECT_ROOT,
        "data/processed/unified_dataset/validation_metadata.csv"
    ),
    "test": os.path.join(
        PROJECT_ROOT,
        "data/processed/unified_dataset/test_metadata.csv"
    )
}

PREDICTION_FILES = {
    "validation": os.path.join(
        PROJECT_ROOT,
        "outputs/ast_exp4/reports/patient_134_audit/device_audit/validation_predictions_exp4.csv"
    ),
    "test": os.path.join(
        PROJECT_ROOT,
        "outputs/ast_exp4/reports/test_predictions.csv"
    )
}

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/sound_class_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_split(split):
    metadata_path = SPLIT_FILES[split]
    prediction_path = PREDICTION_FILES[split]

    if not os.path.exists(metadata_path):
        print(f"ERROR: Missing metadata: {metadata_path}")
        sys.exit(1)

    if not os.path.exists(prediction_path):
        print(f"ERROR: Missing predictions: {prediction_path}")
        sys.exit(1)

    metadata = pd.read_csv(metadata_path)
    predictions = pd.read_csv(prediction_path)

    print()
    print(f"{split.capitalize()} metadata: {len(metadata)} samples")
    print(f"{split.capitalize()} predictions: {len(predictions)} samples")

    if "sample_id" not in metadata.columns:
        print("ERROR: sample_id missing from metadata")
        print(f"Available metadata columns: {list(metadata.columns)}")
        sys.exit(1)

    if "sample_id" not in predictions.columns:
        print("ERROR: sample_id missing from predictions")
        print(f"Available prediction columns: {list(predictions.columns)}")
        sys.exit(1)

    if "predicted_label" in predictions.columns:
        prediction_column = "predicted_label"
    elif "predicted_class" in predictions.columns:
        prediction_column = "predicted_class"
    else:
        print("ERROR: No prediction column found")
        print(f"Available prediction columns: {list(predictions.columns)}")
        sys.exit(1)

    predictions_small = predictions[
        [
            "sample_id",
            prediction_column
        ]
    ].copy()

    predictions_small = predictions_small.rename(
        columns={
            prediction_column: "predicted_label"
        }
    )

    predictions_small = predictions_small.drop_duplicates(
        subset=["sample_id"]
    )

    merged = metadata.merge(
        predictions_small,
        on="sample_id",
        how="inner"
    )

    if len(merged) != len(metadata):
        print(
            f"WARNING: Only {len(merged)} of "
            f"{len(metadata)} metadata samples matched predictions"
        )

    merged["correct"] = (
        merged["class_name"] ==
        merged["predicted_label"]
    )

    merged["split"] = split

    return merged

def analyze_sound_class(df):
    print()
    print("COPD PERFORMANCE BY SOUND CLASS")

    summary_rows = []

    for sound_class, group in df.groupby("sound_class"):
        correct = group["correct"].sum()

        summary_rows.append({
            "sound_class": sound_class,
            "samples": len(group),
            "correct": int(correct),
            "errors": int(len(group) - correct),
            "accuracy": correct / len(group)
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "sound_class"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return summary

def analyze_predictions(df):
    print()
    print("COPD PREDICTION DISTRIBUTION BY SOUND CLASS")

    table = pd.crosstab(
        df["sound_class"],
        df["predicted_label"]
    )

    print(table.to_string())

    return table

def analyze_error_pairs(df):
    print()
    print("COPD ERROR PAIRS BY SOUND CLASS")

    errors = df[~df["correct"]].copy()

    if len(errors) == 0:
        print("No errors.")
        return pd.DataFrame()

    error_pairs = (
        errors
        .groupby(
            [
                "sound_class",
                "class_name",
                "predicted_label"
            ]
        )
        .size()
        .reset_index(name="error_count")
        .sort_values(
            "error_count",
            ascending=False
        )
    )

    print(error_pairs.to_string(index=False))

    return error_pairs

def analyze_patient_sound_class(df):
    print()
    print("PATIENT LEVEL SOUND CLASS PERFORMANCE")

    rows = []

    for patient, group in df.groupby("patient_group"):
        row = {
            "patient_group": patient,
            "split": group["split"].iloc[0],
            "cycles": len(group),
            "correct": int(group["correct"].sum()),
            "accuracy": group["correct"].mean()
        }

        sound_counts = (
            group["sound_class"]
            .value_counts()
            .to_dict()
        )

        for sound_class, count in sound_counts.items():
            row[
                f"sound_class_{sound_class}_count"
            ] = count

            row[
                f"sound_class_{sound_class}_percentage"
            ] = count / len(group) * 100

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        [
            "accuracy",
            "cycles"
        ]
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return summary

def analyze_device_sound_class(df):
    print()
    print("DEVICE AND SOUND CLASS PERFORMANCE")

    if "audio_path" not in df.columns:
        return pd.DataFrame()

    def extract_device(path):
        if pd.isna(path):
            return "UNKNOWN"

        filename = os.path.basename(str(path))

        if filename.lower().endswith(".wav"):
            filename = filename[:-4]

        parts = filename.split("_")

        if len(parts) >= 5:
            return parts[-1]

        return "UNKNOWN"

    data = df.copy()

    data["device"] = data["audio_path"].apply(
        extract_device
    )

    rows = []

    for (
        device,
        sound_class
    ), group in data.groupby(
        [
            "device",
            "sound_class"
        ]
    ):
        rows.append({
            "device": device,
            "sound_class": sound_class,
            "samples": len(group),
            "correct": int(group["correct"].sum()),
            "errors": int((~group["correct"]).sum()),
            "accuracy": group["correct"].mean()
        })

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return summary

def analyze_134_199_141(df):
    print()
    print("PATIENT 134 VS 199 VS 141")

    target_patients = [
        "ICBHI_134",
        "ICBHI_199",
        "ICBHI_141"
    ]

    target = df[
        df["patient_group"].isin(target_patients)
    ].copy()

    if len(target) == 0:
        print("No target patients in this split.")
        return pd.DataFrame()

    summary = (
        target
        .groupby(
            [
                "patient_group",
                "sound_class"
            ]
        )
        .agg(
            samples=("sample_id", "size"),
            correct=("correct", "sum")
        )
        .reset_index()
    )

    summary["errors"] = (
        summary["samples"] -
        summary["correct"]
    )

    summary["accuracy"] = (
        summary["correct"] /
        summary["samples"]
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return summary

def analyze_cycle_duration(df):
    print()
    print("COPD CYCLE DURATION BY SOUND CLASS")

    if (
        "start_time" not in df.columns or
        "end_time" not in df.columns
    ):
        print("start_time or end_time missing.")
        return pd.DataFrame()

    data = df.copy()

    data["cycle_duration"] = (
        pd.to_numeric(
            data["end_time"],
            errors="coerce"
        ) -
        pd.to_numeric(
            data["start_time"],
            errors="coerce"
        )
    )

    summary = (
        data
        .groupby("sound_class")["cycle_duration"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max"
            ]
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return summary

def main():
    validation = load_split("validation")
    test = load_split("test")

    combined = pd.concat(
        [
            validation,
            test
        ],
        ignore_index=True
    )

    icbhi_copd = combined[
        (combined["source"] == "ICBHI") &
        (combined["class_name"] == "COPD")
    ].copy()

    print()
    print("ICBHI COPD VALIDATION + TEST")
    print(
        f"Total COPD cycles: {len(icbhi_copd)}"
    )

    validation_copd = validation[
        (validation["source"] == "ICBHI") &
        (validation["class_name"] == "COPD")
    ]

    test_copd = test[
        (test["source"] == "ICBHI") &
        (test["class_name"] == "COPD")
    ]

    print(
        f"Validation COPD cycles: {len(validation_copd)}"
    )

    print(
        f"Test COPD cycles: {len(test_copd)}"
    )

    print()
    print("OVERALL SOUND CLASS DISTRIBUTION")

    print(
        icbhi_copd["sound_class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    validation_summary = analyze_sound_class(
        validation_copd
    )

    test_summary = analyze_sound_class(
        test_copd
    )

    combined_summary = analyze_sound_class(
        icbhi_copd
    )

    validation_predictions = analyze_predictions(
        validation_copd
    )

    test_predictions = analyze_predictions(
        test_copd
    )

    combined_predictions = analyze_predictions(
        icbhi_copd
    )

    validation_errors = analyze_error_pairs(
        validation_copd
    )

    test_errors = analyze_error_pairs(
        test_copd
    )

    combined_errors = analyze_error_pairs(
        icbhi_copd
    )

    patient_summary = analyze_patient_sound_class(
        icbhi_copd
    )

    device_summary = analyze_device_sound_class(
        icbhi_copd
    )

    duration_summary = analyze_cycle_duration(
        icbhi_copd
    )

    validation_targets = analyze_134_199_141(
        validation_copd
    )

    test_targets = analyze_134_199_141(
        test_copd
    )

    icbhi_copd.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icbhi_copd_validation_test_predictions_with_sound_class.csv"
        ),
        index=False
    )

    validation_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_sound_class_performance.csv"
        ),
        index=False
    )

    test_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_sound_class_performance.csv"
        ),
        index=False
    )

    combined_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "combined_sound_class_performance.csv"
        ),
        index=False
    )

    validation_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_sound_class_predictions.csv"
        )
    )

    test_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_sound_class_predictions.csv"
        )
    )

    combined_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "combined_sound_class_predictions.csv"
        )
    )

    validation_errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_sound_class_errors.csv"
        ),
        index=False
    )

    test_errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_sound_class_errors.csv"
        ),
        index=False
    )

    combined_errors.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "combined_sound_class_errors.csv"
        ),
        index=False
    )

    patient_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "patient_sound_class_performance.csv"
        ),
        index=False
    )

    device_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "device_sound_class_performance.csv"
        ),
        index=False
    )

    duration_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "cycle_duration_by_sound_class.csv"
        ),
        index=False
    )

    validation_targets.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_134_199_141_sound_class.csv"
        ),
        index=False
    )

    test_targets.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_134_199_141_sound_class.csv"
        ),
        index=False
    )

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)

    print()
    print("Generated files:")

    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(filename)

if __name__ == "__main__":
    main()