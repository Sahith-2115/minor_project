import os
import sys
import pandas as pd

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/device_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = [
    "train",
    "validation",
    "test"
]

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

def load_split(split):
    path = os.path.join(
        DATASET_DIR,
        f"{split}_metadata.csv"
    )

    if not os.path.exists(path):
        print(f"ERROR: Missing file: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    df["split"] = split

    df["device"] = df["audio_path"].apply(
        extract_device
    )

    return df

def main():
    print("Loading unified dataset splits")

    frames = []

    for split in SPLITS:
        df = load_split(split)

        print(
            f"{split}: {len(df)} samples"
        )

        frames.append(df)

    metadata = pd.concat(
        frames,
        ignore_index=True
    )

    icbhi = metadata[
        metadata["source"] == "ICBHI"
    ].copy()

    print()
    print("Complete ICBHI unified dataset")
    print(
        f"Total cycles: {len(icbhi)}"
    )

    print(
        f"Patients: "
        f"{icbhi['patient_group'].nunique()}"
    )

    print(
        f"Devices: "
        f"{icbhi['device'].nunique()}"
    )

    print()
    print("Complete ICBHI device distribution")

    print(
        icbhi["device"]
        .value_counts()
        .to_string()
    )

    print()
    print("Device by disease class")

    device_class = pd.crosstab(
        icbhi["device"],
        icbhi["class_name"]
    )

    print(
        device_class.to_string()
    )

    device_class.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "full_icbhi_device_by_class.csv"
        )
    )

    print()
    print("Device by split")

    device_split = pd.crosstab(
        icbhi["device"],
        icbhi["split"]
    )

    print(
        device_split.to_string()
    )

    device_split.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "full_icbhi_device_by_split.csv"
        )
    )

    print()
    print("Device by class and split")

    device_class_split = (
        icbhi
        .groupby(
            [
                "device",
                "split",
                "class_name"
            ]
        )
        .size()
        .reset_index(
            name="cycle_count"
        )
    )

    print(
        device_class_split.to_string(
            index=False
        )
    )

    device_class_split.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "full_icbhi_device_class_split.csv"
        ),
        index=False
    )

    print()
    print("Patient count by device")

    device_patients = (
        icbhi
        .groupby("device")
        ["patient_group"]
        .nunique()
        .reset_index(
            name="patient_count"
        )
        .sort_values(
            "patient_count",
            ascending=False
        )
    )

    print(
        device_patients.to_string(
            index=False
        )
    )

    device_patients.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "full_icbhi_device_patient_counts.csv"
        ),
        index=False
    )

    print()
    print("Patient groups by device")

    device_patient_list = (
        icbhi
        .groupby("device")
        ["patient_group"]
        .unique()
        .reset_index()
    )

    device_patient_list[
        "patient_groups"
    ] = device_patient_list[
        "patient_group"
    ].apply(
        lambda x: ", ".join(
            sorted(x)
        )
    )

    device_patient_list = (
        device_patient_list
        .drop(columns=["patient_group"])
    )

    print(
        device_patient_list.to_string(
            index=False
        )
    )

    device_patient_list.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "full_icbhi_device_patient_groups.csv"
        ),
        index=False
    )

    print()
    print("=" * 70)
    print("TEST DEVICE TRAINING EXPOSURE")
    print("=" * 70)

    test_icbhi = icbhi[
        icbhi["split"] == "test"
    ].copy()

    train_icbhi = icbhi[
        icbhi["split"] == "train"
    ].copy()

    test_devices = sorted(
        test_icbhi["device"].unique()
    )

    exposure_rows = []

    for device in test_devices:
        train_device = train_icbhi[
            train_icbhi["device"] == device
        ]

        test_device = test_icbhi[
            test_icbhi["device"] == device
        ]

        row = {
            "device": device,
            "test_cycles": len(test_device),
            "test_patients": test_device[
                "patient_group"
            ].nunique(),
            "train_cycles_same_device": len(
                train_device
            ),
            "train_patients_same_device": (
                train_device[
                    "patient_group"
                ].nunique()
            ),
            "seen_in_training": len(
                train_device
            ) > 0
        }

        exposure_rows.append(row)

    exposure = pd.DataFrame(
        exposure_rows
    )

    print(
        exposure.to_string(
            index=False
        )
    )

    exposure.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_device_training_exposure.csv"
        ),
        index=False
    )

    print()
    print("=" * 70)
    print("LittC2SE DETAILED ANALYSIS")
    print("=" * 70)

    litt = icbhi[
        icbhi["device"] == "LittC2SE"
    ].copy()

    if len(litt) == 0:
        print(
            "No LittC2SE recordings found."
        )
    else:
        print()
        print(
            f"LittC2SE total cycles: {len(litt)}"
        )

        print(
            f"LittC2SE patients: "
            f"{litt['patient_group'].nunique()}"
        )

        print()
        print("LittC2SE by split")

        print(
            pd.crosstab(
                litt["split"],
                litt["class_name"]
            ).to_string()
        )

        print()
        print("LittC2SE patient groups")

        litt_patients = (
            litt
            .groupby(
                [
                    "patient_group",
                    "split",
                    "class_name"
                ]
            )
            .size()
            .reset_index(
                name="cycle_count"
            )
        )

        print(
            litt_patients.to_string(
                index=False
            )
        )

        litt_patients.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "littc2se_patient_analysis.csv"
            ),
            index=False
        )

    print()
    print("=" * 70)
    print("PATIENT 134 TRAINING EXPOSURE")
    print("=" * 70)

    patient_134 = icbhi[
        icbhi["patient_group"] == "ICBHI_134"
    ]

    print(
        patient_134[
            [
                "patient_group",
                "split",
                "class_name",
                "device",
                "audio_path"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    print()
    print("Other COPD patients using LittC2SE")

    other_litt_copd = icbhi[
        (icbhi["device"] == "LittC2SE") &
        (icbhi["class_name"] == "COPD") &
        (icbhi["patient_group"] != "ICBHI_134")
    ]

    if len(other_litt_copd) == 0:
        print(
            "No other COPD patient uses LittC2SE."
        )
    else:
        print(
            other_litt_copd[
                [
                    "patient_group",
                    "split",
                    "class_name",
                    "device"
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
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