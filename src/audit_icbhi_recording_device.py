import os
import sys
import pandas as pd

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

TEST_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/test_metadata.csv"
)

PREDICTIONS = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/device_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def extract_recording_protocol(path):
    if pd.isna(path):
        return "UNKNOWN"

    filename = os.path.basename(str(path))

    if filename.lower().endswith(".wav"):
        filename = filename[:-4]

    parts = filename.split("_")

    if len(parts) >= 4:
        return parts[-2]

    return "UNKNOWN"

def main():
    print(f"Loading metadata: {TEST_METADATA}")

    if not os.path.exists(TEST_METADATA):
        print("ERROR: Test metadata not found.")
        sys.exit(1)

    metadata = pd.read_csv(TEST_METADATA)

    required_columns = [
        "audio_path",
        "source",
        "class_name",
        "patient_group"
    ]

    missing = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    icbhi = metadata[
        metadata["source"] == "ICBHI"
    ].copy()

    icbhi["device"] = icbhi["audio_path"].apply(
        extract_device
    )

    icbhi["protocol"] = icbhi["audio_path"].apply(
        extract_recording_protocol
    )

    print()
    print("ICBHI dataset")
    print(f"Total cycles: {len(icbhi)}")
    print(f"Patient groups: {icbhi['patient_group'].nunique()}")
    print(f"Devices: {icbhi['device'].nunique()}")
    print(f"Protocols: {icbhi['protocol'].nunique()}")

    print()
    print("Device distribution")
    print(
        icbhi["device"]
        .value_counts()
        .to_string()
    )

    print()
    print("Protocol distribution")
    print(
        icbhi["protocol"]
        .value_counts()
        .to_string()
    )

    device_class = pd.crosstab(
        icbhi["device"],
        icbhi["class_name"]
    )

    device_class.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icbhi_device_by_class.csv"
        )
    )

    print()
    print("Device by disease class")
    print(device_class.to_string())

    device_patients = (
        icbhi.groupby("device")["patient_group"]
        .nunique()
        .sort_values(ascending=False)
    )

    device_patients.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icbhi_device_patient_counts.csv"
        )
    )

    print()
    print("Patients per device")
    print(device_patients.to_string())

    device_recordings = (
        icbhi.groupby("device")["audio_path"]
        .nunique()
        .sort_values(ascending=False)
    )

    device_recordings.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icbhi_device_recording_counts.csv"
        )
    )

    print()
    print("Audio recordings per device")
    print(device_recordings.to_string())

    protocol_class = pd.crosstab(
        icbhi["protocol"],
        icbhi["class_name"]
    )

    protocol_class.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icbhi_protocol_by_class.csv"
        )
    )

    print()
    print("Protocol by disease class")
    print(protocol_class.to_string())

    print()
    print("Test set device analysis")

    test = icbhi.copy()

    if os.path.exists(PREDICTIONS):
        predictions = pd.read_csv(PREDICTIONS)

        print(
            f"Loaded predictions from: {PREDICTIONS}"
        )

        metadata_id = None

        for column in [
            "sample_id",
            "id"
        ]:
            if column in test.columns:
                metadata_id = column
                break

        prediction_id = None

        for column in [
            "sample_id",
            "id"
        ]:
            if column in predictions.columns:
                prediction_id = column
                break

        predicted_label = None

        for column in [
            "predicted_class",
            "prediction",
            "predicted_label"
        ]:
            if column in predictions.columns:
                predicted_label = column
                break

        if (
            metadata_id is not None and
            prediction_id is not None and
            predicted_label is not None
        ):
            pred_small = predictions[
                [
                    prediction_id,
                    predicted_label
                ]
            ].copy()

            pred_small = pred_small.rename(
                columns={
                    prediction_id: metadata_id,
                    predicted_label: "predicted_class"
                }
            )

            test = test.merge(
                pred_small,
                on=metadata_id,
                how="left"
            )

            test["correct"] = (
                test["class_name"] ==
                test["predicted_class"]
            )

            test["error_pair"] = (
                test["class_name"].astype(str)
                + " -> "
                + test["predicted_class"].astype(str)
            )

    test_device_class = pd.crosstab(
        test["device"],
        test["class_name"]
    )

    test_device_class.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_device_by_class.csv"
        )
    )

    print()
    print("Test device by class")
    print(test_device_class.to_string())

    if "correct" in test.columns:
        device_performance = (
            test.groupby("device")
            .agg(
                cycles=("class_name", "size"),
                patients=("patient_group", "nunique"),
                correct=("correct", "sum")
            )
        )

        device_performance["accuracy"] = (
            device_performance["correct"] /
            device_performance["cycles"]
        )

        device_performance = (
            device_performance
            .sort_values(
                "cycles",
                ascending=False
            )
        )

        device_performance.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "test_device_performance.csv"
            )
        )

        print()
        print("Test performance by device")
        print(
            device_performance.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

        error_pairs = (
            test[~test["correct"]]
            .groupby(
                [
                    "device",
                    "class_name",
                    "predicted_class"
                ]
            )
            .size()
            .reset_index(
                name="error_count"
            )
            .sort_values(
                "error_count",
                ascending=False
            )
        )

        error_pairs.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "test_device_error_pairs.csv"
            ),
            index=False
        )

        print()
        print("Errors by device")
        print(
            error_pairs.to_string(
                index=False
            )
        )

        copd_errors = test[
            (test["class_name"] == "COPD") &
            (~test["correct"])
        ].copy()

        print()
        print("COPD errors by device")

        copd_device_errors = (
            copd_errors
            .groupby(
                [
                    "device",
                    "predicted_class"
                ]
            )
            .size()
            .reset_index(
                name="error_count"
            )
            .sort_values(
                "error_count",
                ascending=False
            )
        )

        print(
            copd_device_errors.to_string(
                index=False
            )
        )

        copd_device_errors.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "copd_errors_by_device.csv"
            ),
            index=False
        )

        print()
        print("COPD performance by device")

        copd_test = test[
            test["class_name"] == "COPD"
        ].copy()

        copd_device_performance = (
            copd_test.groupby("device")
            .agg(
                cycles=("class_name", "size"),
                patients=("patient_group", "nunique"),
                correct=("correct", "sum")
            )
        )

        copd_device_performance["accuracy"] = (
            copd_device_performance["correct"] /
            copd_device_performance["cycles"]
        )

        copd_device_performance.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "copd_performance_by_device.csv"
            )
        )

        print(
            copd_device_performance.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

        patient_134 = test[
            test["patient_group"] == "ICBHI_134"
        ].copy()

        print()
        print("Patient 134 device information")

        if len(patient_134) > 0:
            print(
                patient_134[
                    [
                        "patient_group",
                        "device",
                        "protocol",
                        "class_name"
                    ]
                ]
                .drop_duplicates()
                .to_string(index=False)
            )

            print()
            print("Patient 134 predictions")

            print(
                patient_134[
                    [
                        "class_name",
                        "predicted_class"
                    ]
                ]
                .value_counts()
                .to_string()
            )

            print()
            print("Patient 134 device and protocol")

            print(
                "Device:",
                patient_134["device"].iloc[0]
            )

            print(
                "Protocol:",
                patient_134["protocol"].iloc[0]
            )

    device_patient = (
        icbhi[
            icbhi["class_name"].isin(
                [
                    "COPD",
                    "Healthy",
                    "Pneumonia"
                ]
            )
        ]
        .groupby(
            [
                "device",
                "class_name"
            ]
        )["patient_group"]
        .nunique()
        .reset_index(
            name="patient_count"
        )
    )

    device_patient.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "device_by_disease_patient_counts.csv"
        ),
        index=False
    )

    print()
    print("Disease patient counts by device")
    print(
        device_patient.to_string(
            index=False
        )
    )

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("Generated files:")

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):
        print(filename)

if __name__ == "__main__":
    main()