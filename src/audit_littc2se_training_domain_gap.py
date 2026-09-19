import os
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

TRAIN_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/train_metadata.csv"
)

VALIDATION_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/validation_metadata.csv"
)

TEST_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/test_metadata.csv"
)

VALIDATION_PREDICTIONS = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/device_audit/validation_predictions_exp4.csv"
)

TEST_PREDICTIONS = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/training_domain_gap"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "rms",
    "peak",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff"
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

def load_audio_features(path):
    try:
        audio, sr = librosa.load(
            path,
            sr=None,
            mono=True
        )

        if len(audio) == 0:
            return None

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        zcr = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    audio
                )
            )
        )

        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=audio,
                    sr=sr
                )
            )
        )

        bandwidth = float(
            np.mean(
                librosa.feature.spectral_bandwidth(
                    y=audio,
                    sr=sr
                )
            )
        )

        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=audio,
                    sr=sr,
                    roll_percent=0.85
                )
            )
        )

        return {
            "sample_rate": sr,
            "audio_duration": len(audio) / sr,
            "rms": rms,
            "peak": peak,
            "zcr": zcr,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff
        }

    except Exception as e:
        print(
            f"ERROR loading {path}: {e}"
        )
        return None

def load_metadata():
    train = pd.read_csv(
        TRAIN_METADATA
    )

    validation = pd.read_csv(
        VALIDATION_METADATA
    )

    test = pd.read_csv(
        TEST_METADATA
    )

    validation_predictions = pd.read_csv(
        VALIDATION_PREDICTIONS
    )

    test_predictions = pd.read_csv(
        TEST_PREDICTIONS
    )

    if "predicted_class" in validation_predictions.columns:
        validation_prediction_column = "predicted_class"
    elif "predicted_label" in validation_predictions.columns:
        validation_prediction_column = "predicted_label"
    else:
        print(
            "ERROR: Validation prediction column not found."
        )
        print(
            list(validation_predictions.columns)
        )
        sys.exit(1)

    if "predicted_label" in test_predictions.columns:
        test_prediction_column = "predicted_label"
    elif "predicted_class" in test_predictions.columns:
        test_prediction_column = "predicted_class"
    else:
        print(
            "ERROR: Test prediction column not found."
        )
        print(
            list(test_predictions.columns)
        )
        sys.exit(1)

    validation_predictions = (
        validation_predictions[
            [
                "sample_id",
                validation_prediction_column
            ]
        ]
        .rename(
            columns={
                validation_prediction_column:
                "predicted_label"
            }
        )
        .drop_duplicates(
            "sample_id"
        )
    )

    test_predictions = (
        test_predictions[
            [
                "sample_id",
                test_prediction_column
            ]
        ]
        .rename(
            columns={
                test_prediction_column:
                "predicted_label"
            }
        )
        .drop_duplicates(
            "sample_id"
        )
    )

    validation = validation.merge(
        validation_predictions,
        on="sample_id",
        how="left"
    )

    test = test.merge(
        test_predictions,
        on="sample_id",
        how="left"
    )

    validation["correct"] = (
        validation["class_name"] ==
        validation["predicted_label"]
    )

    test["correct"] = (
        test["class_name"] ==
        test["predicted_label"]
    )

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    combined = pd.concat(
        [
            train,
            validation,
            test
        ],
        ignore_index=True
    )

    return combined

def select_littc2se_copd_class0(data):
    data = data[
        (data["source"] == "ICBHI") &
        (data["class_name"] == "COPD") &
        (data["sound_class"] == 0)
    ].copy()

    data["device"] = data[
        "audio_path"
    ].apply(
        extract_device
    )

    data = data[
        data["device"] == "LittC2SE"
    ].copy()

    return data

def extract_recording_features(data):
    recordings = (
        data[
            [
                "audio_path",
                "patient_group",
                "split",
                "correct"
            ]
        ]
        .drop_duplicates(
            subset=["audio_path"]
        )
        .copy()
    )

    rows = []

    print()
    print(
        f"Extracting features from "
        f"{len(recordings)} recordings..."
    )

    for _, row in tqdm(
        recordings.iterrows(),
        total=len(recordings),
        desc="Acoustic analysis"
    ):
        features = load_audio_features(
            row["audio_path"]
        )

        if features is None:
            continue

        result = {
            "audio_path": row["audio_path"],
            "patient_group": row["patient_group"],
            "split": row["split"],
            "correct": row["correct"]
        }

        result.update(features)

        rows.append(result)

    return pd.DataFrame(rows)

def calculate_group_statistics(features):
    groups = []

    for (
        split,
        patient
    ), group in features.groupby(
        [
            "split",
            "patient_group"
        ]
    ):
        row = {
            "split": split,
            "patient_group": patient,
            "recordings": len(group)
        }

        for feature in FEATURE_COLUMNS:
            row[
                f"{feature}_mean"
            ] = group[feature].mean()

            row[
                f"{feature}_median"
            ] = group[feature].median()

            row[
                f"{feature}_std"
            ] = group[feature].std()

        groups.append(row)

    return pd.DataFrame(groups)

def print_dataset_summary(data):
    print()
    print(
        "LittC2SE COPD CLASS 0 DATASET SUMMARY"
    )

    print(
        data.groupby(
            [
                "split",
                "patient_group"
            ]
        ).size().to_string()
    )

    print()
    print(
        "Cycles by split:"
    )

    print(
        data["split"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Unique recordings by split:"
    )

    print(
        data.groupby(
            "split"
        )["audio_path"]
        .nunique()
        .to_string()
    )

def calculate_reference_profiles(
    features
):
    print()
    print(
        "REFERENCE PATIENT PROFILES"
    )

    target_patients = [
        "ICBHI_134",
        "ICBHI_199"
    ]

    rows = []

    for patient in target_patients:
        group = features[
            features["patient_group"] == patient
        ]

        if len(group) == 0:
            continue

        row = {
            "patient_group": patient,
            "split": group["split"].iloc[0],
            "recordings": len(group)
        }

        for feature in FEATURE_COLUMNS:
            row[feature] = group[feature].mean()

        rows.append(row)

    reference = pd.DataFrame(rows)

    if len(reference) > 0:
        print(
            reference.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    return reference

def calculate_training_distance(
    features
):
    print()
    print(
        "DISTANCE FROM PROBLEMATIC PATIENTS"
    )

    training = features[
        features["split"] == "train"
    ].copy()

    references = {}

    for patient in [
        "ICBHI_134",
        "ICBHI_199"
    ]:
        reference = features[
            features["patient_group"] == patient
        ]

        if len(reference) == 0:
            continue

        references[patient] = {
            feature: reference[feature].mean()
            for feature in FEATURE_COLUMNS
        }

    if len(training) == 0:
        return pd.DataFrame()

    rows = []

    training_patient_features = (
        training
        .groupby("patient_group")
        [FEATURE_COLUMNS]
        .mean()
        .reset_index()
    )

    for _, row in training_patient_features.iterrows():
        result = {
            "patient_group":
            row["patient_group"]
        }

        for reference_name, reference_values in references.items():
            distance_values = []

            for feature in FEATURE_COLUMNS:
                training_values = features[
                    (
                        features["patient_group"] ==
                        row["patient_group"]
                    ) &
                    (
                        features["split"] ==
                        "train"
                    )
                ][feature]

                if len(training_values) == 0:
                    continue

                all_training = training[
                    feature
                ]

                std = all_training.std()

                if std == 0 or pd.isna(std):
                    std = 1.0

                distance = (
                    row[feature] -
                    reference_values[feature]
                ) / std

                distance_values.append(
                    distance ** 2
                )

            if distance_values:
                result[
                    f"distance_to_{reference_name}"
                ] = np.sqrt(
                    np.mean(
                        distance_values
                    )
                )

        rows.append(result)

    distances = pd.DataFrame(rows)

    return distances

def nearest_training_recordings(
    features,
    reference_patient,
    top_n=15
):
    print()
    print(
        f"NEAREST TRAINING RECORDINGS TO "
        f"{reference_patient}"
    )

    reference = features[
        features["patient_group"] ==
        reference_patient
    ]

    training = features[
        features["split"] == "train"
    ].copy()

    if len(reference) == 0 or len(training) == 0:
        print(
            "Reference or training data missing."
        )
        return pd.DataFrame()

    reference_vector = reference[
        FEATURE_COLUMNS
    ].mean()

    means = training[
        FEATURE_COLUMNS
    ].mean()

    stds = training[
        FEATURE_COLUMNS
    ].std()

    stds = stds.replace(
        0,
        1.0
    ).fillna(1.0)

    normalized_training = (
        training[FEATURE_COLUMNS] -
        means
    ) / stds

    normalized_reference = (
        reference_vector -
        means
    ) / stds

    distances = np.sqrt(
        np.sum(
            np.square(
                normalized_training -
                normalized_reference
            ),
            axis=1
        )
    )

    training = training.copy()

    training[
        f"distance_to_{reference_patient}"
    ] = distances.values

    result = training.sort_values(
        f"distance_to_{reference_patient}"
    ).head(top_n)

    columns = [
        "patient_group",
        "audio_path",
        "split",
        f"distance_to_{reference_patient}"
    ]

    result = result[columns]

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return result

def compare_train_and_problem_patients(
    features
):
    print()
    print(
        "TRAINING VS PROBLEMATIC PATIENTS"
    )

    training = features[
        features["split"] == "train"
    ]

    problem = features[
        features["patient_group"].isin(
            [
                "ICBHI_134",
                "ICBHI_199"
            ]
        )
    ]

    rows = []

    for feature in FEATURE_COLUMNS:
        rows.append({
            "feature": feature,
            "train_mean": training[feature].mean(),
            "train_median": training[feature].median(),
            "problem_mean": problem[feature].mean(),
            "problem_median": problem[feature].median()
        })

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return summary

def main():
    print(
        "Loading metadata and predictions..."
    )

    data = load_metadata()

    print(
        f"Total unified samples: {len(data)}"
    )

    littc2se = select_littc2se_copd_class0(
        data
    )

    print_dataset_summary(
        littc2se
    )

    features = extract_recording_features(
        littc2se
    )

    if len(features) == 0:
        print(
            "ERROR: No audio features extracted."
        )
        sys.exit(1)

    patient_profiles = calculate_group_statistics(
        features
    )

    reference_profiles = calculate_reference_profiles(
        features
    )

    training_distances = calculate_training_distance(
        features
    )

    compare_summary = (
        compare_train_and_problem_patients(
            features
        )
    )

    nearest_134 = nearest_training_recordings(
        features,
        "ICBHI_134",
        top_n=15
    )

    nearest_199 = nearest_training_recordings(
        features,
        "ICBHI_199",
        top_n=15
    )

    littc2se.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_all_cycles.csv"
        ),
        index=False
    )

    features.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_all_audio_features.csv"
        ),
        index=False
    )

    patient_profiles.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "littc2se_copd_class0_patient_profiles.csv"
        ),
        index=False
    )

    reference_profiles.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "problematic_patient_reference_profiles.csv"
        ),
        index=False
    )

    training_distances.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "training_patient_distances.csv"
        ),
        index=False
    )

    compare_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "training_vs_problem_patient_features.csv"
        ),
        index=False
    )

    nearest_134.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "nearest_training_recordings_to_134.csv"
        ),
        index=False
    )

    nearest_199.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "nearest_training_recordings_to_199.csv"
        ),
        index=False
    )

    print()
    print(
        "OUTPUT DIRECTORY"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Generated files:"
    )

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):
        print(filename)

if __name__ == "__main__":
    main()