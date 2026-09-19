import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

TEST_PREDICTIONS = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/test_predictions.csv"
)

TEST_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/test_metadata.csv"
)

VALIDATION_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/validation_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports"
)

CLASS_NAMES = [
    "Asthma",
    "Bronchitis",
    "COPD",
    "Healthy",
    "Pneumonia"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("AST EXPERIMENT 4 ERROR ANALYSIS")
print("Loading prediction and metadata files...")

if not os.path.exists(TEST_PREDICTIONS):
    raise FileNotFoundError(
        f"Prediction file not found: {TEST_PREDICTIONS}"
    )

if not os.path.exists(TEST_METADATA):
    raise FileNotFoundError(
        f"Test metadata not found: {TEST_METADATA}"
    )

if not os.path.exists(VALIDATION_METADATA):
    raise FileNotFoundError(
        f"Validation metadata not found: {VALIDATION_METADATA}"
    )

pred_df = pd.read_csv(TEST_PREDICTIONS)
test_df = pd.read_csv(TEST_METADATA)
val_df = pd.read_csv(VALIDATION_METADATA)

print(f"Test prediction rows: {len(pred_df)}")
print(f"Test metadata rows: {len(test_df)}")
print(f"Validation metadata rows: {len(val_df)}")

print()
print("Prediction file columns:")
print(pred_df.columns.tolist())

print()
print("Test metadata columns:")
print(test_df.columns.tolist())

required_prediction_columns = [
    "true_label",
    "predicted_label",
    "correct"
]

for column in required_prediction_columns:
    if column not in pred_df.columns:
        raise ValueError(
            f"Required prediction column missing: {column}"
        )

for dataframe_name, dataframe in [
    ("test predictions", pred_df),
    ("test metadata", test_df),
    ("validation metadata", val_df)
]:
    if "class_name" not in dataframe.columns:
        raise ValueError(
            f"class_name column missing from {dataframe_name}"
        )

def normalize_source(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value)

    if "ICBHI" in value.upper():
        return "ICBHI"

    if "ASTHMA" in value.upper():
        return "AsthmaV2"

    return value

def get_source_column(dataframe):
    possible_columns = [
        "source",
        "dataset_source",
        "source_dataset"
    ]

    for column in possible_columns:
        if column in dataframe.columns:
            return column

    return None

def get_patient_column(dataframe):
    possible_columns = [
        "patient_group",
        "patient_id",
        "patient",
        "patient_group_id"
    ]

    for column in possible_columns:
        if column in dataframe.columns:
            return column

    return None

source_column = get_source_column(pred_df)

if source_column is None:
    source_column = get_source_column(test_df)

patient_column = get_patient_column(pred_df)

if patient_column is None:
    patient_column = get_patient_column(test_df)

print()
print(f"Source column: {source_column}")
print(f"Patient column: {patient_column}")

if source_column is None:
    print()
    print("WARNING: No source column was found.")
    print("Source level analysis cannot be performed.")

if patient_column is None:
    print()
    print("WARNING: No patient group column was found.")
    print("Patient level analysis cannot be performed.")

if source_column is not None:
    if source_column not in pred_df.columns:
        pred_df[source_column] = test_df[source_column].values

    pred_df["analysis_source"] = pred_df[source_column].apply(
        normalize_source
    )

if patient_column is not None:
    if patient_column not in pred_df.columns:
        pred_df[patient_column] = test_df[patient_column].values

    pred_df["analysis_patient_group"] = (
        pred_df[patient_column].astype(str)
    )

print()
print("OVERALL TEST PERFORMANCE")

true_labels = pred_df["true_label"].astype(str)
predicted_labels = pred_df["predicted_label"].astype(str)

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=CLASS_NAMES,
        average="weighted",
        zero_division=0
    )
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print()
print("TEST CLASS DISTRIBUTION")

test_class_counts = (
    pred_df["true_label"]
    .value_counts()
    .reindex(CLASS_NAMES)
    .fillna(0)
    .astype(int)
)

print(test_class_counts.to_string())

print()
print("OVERALL CONFUSION MATRIX")

overall_cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=CLASS_NAMES
)

cm_df = pd.DataFrame(
    overall_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

print(cm_df.to_string())

cm_path = os.path.join(
    OUTPUT_DIR,
    "exp4_error_analysis_overall_confusion.csv"
)

cm_df.to_csv(cm_path)

print()
print("TEST ERROR PAIRS")

error_df = pred_df[
    pred_df["true_label"] != pred_df["predicted_label"]
].copy()

error_pairs = (
    error_df
    .groupby(
        ["true_label", "predicted_label"]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False
    )
)

if len(error_pairs) == 0:
    print("No errors found.")

else:
    print(error_pairs.to_string(index=False))

error_pairs_path = os.path.join(
    OUTPUT_DIR,
    "exp4_error_pairs.csv"
)

error_pairs.to_csv(
    error_pairs_path,
    index=False
)

print()
print("CLASS LEVEL TEST PERFORMANCE")

class_precision, class_recall, class_f1, class_support = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=CLASS_NAMES,
        average=None,
        zero_division=0
    )
)

class_results = pd.DataFrame({
    "class": CLASS_NAMES,
    "precision": class_precision,
    "recall": class_recall,
    "f1": class_f1,
    "support": class_support
})

print(class_results.to_string(index=False))

class_results_path = os.path.join(
    OUTPUT_DIR,
    "exp4_error_analysis_class_metrics.csv"
)

class_results.to_csv(
    class_results_path,
    index=False
)

if source_column is not None:

    print()
    print("SOURCE DISTRIBUTION IN TEST SET")

    source_class_table = pd.crosstab(
        pred_df["analysis_source"],
        pred_df["true_label"]
    )

    source_class_table = source_class_table.reindex(
        columns=CLASS_NAMES,
        fill_value=0
    )

    source_class_table["Total"] = (
        source_class_table.sum(axis=1)
    )

    print(source_class_table.to_string())

    source_class_path = os.path.join(
        OUTPUT_DIR,
        "exp4_test_source_class_distribution.csv"
    )

    source_class_table.to_csv(
        source_class_path
    )

    print()
    print("SOURCE LEVEL PERFORMANCE")

    source_results = []

    for source in sorted(
        pred_df["analysis_source"].unique()
    ):

        source_data = pred_df[
            pred_df["analysis_source"] == source
        ]

        y_true = source_data["true_label"].astype(str)
        y_pred = source_data["predicted_label"].astype(str)

        source_accuracy = accuracy_score(
            y_true,
            y_pred
        )

        source_precision, source_recall, source_f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=CLASS_NAMES,
                average="macro",
                zero_division=0
            )
        )

        source_results.append({
            "source": source,
            "samples": len(source_data),
            "accuracy": source_accuracy,
            "macro_precision": source_precision,
            "macro_recall": source_recall,
            "macro_f1": source_f1
        })

        print()
        print(f"Source: {source}")
        print(f"Samples: {len(source_data)}")
        print(f"Accuracy: {source_accuracy:.4f}")
        print(f"Macro Precision: {source_precision:.4f}")
        print(f"Macro Recall: {source_recall:.4f}")
        print(f"Macro F1: {source_f1:.4f}")

        source_cm = confusion_matrix(
            y_true,
            y_pred,
            labels=CLASS_NAMES
        )

        source_cm_df = pd.DataFrame(
            source_cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        )

        print("Confusion Matrix:")
        print(source_cm_df.to_string())

        safe_source = (
            source
            .replace("/", "_")
            .replace(" ", "_")
        )

        source_cm_path = os.path.join(
            OUTPUT_DIR,
            f"exp4_confusion_{safe_source}.csv"
        )

        source_cm_df.to_csv(
            source_cm_path
        )

    source_results_df = pd.DataFrame(
        source_results
    )

    source_results_path = os.path.join(
        OUTPUT_DIR,
        "exp4_source_level_performance.csv"
    )

    source_results_df.to_csv(
        source_results_path,
        index=False
    )

    print()
    print("SOURCE SPECIFIC ERROR PAIRS")

    source_error_rows = []

    for source in sorted(
        pred_df["analysis_source"].unique()
    ):

        source_errors = error_df[
            error_df["analysis_source"] == source
        ]

        if len(source_errors) == 0:
            continue

        pair_counts = (
            source_errors
            .groupby(
                ["true_label", "predicted_label"]
            )
            .size()
            .reset_index(name="count")
        )

        pair_counts["source"] = source

        source_error_rows.append(
            pair_counts
        )

    if source_error_rows:
        source_errors_df = pd.concat(
            source_error_rows,
            ignore_index=True
        )

        source_errors_df = source_errors_df[
            [
                "source",
                "true_label",
                "predicted_label",
                "count"
            ]
        ].sort_values(
            "count",
            ascending=False
        )

        print(
            source_errors_df.to_string(
                index=False
            )
        )

        source_errors_path = os.path.join(
            OUTPUT_DIR,
            "exp4_source_error_pairs.csv"
        )

        source_errors_df.to_csv(
            source_errors_path,
            index=False
        )

print()
print("VALIDATION AND TEST COMPOSITION")

def composition_table(dataframe, dataset_name):

    table = pd.crosstab(
        dataframe.get(
            source_column,
            pd.Series(
                ["Unknown"] * len(dataframe)
            )
        ).apply(normalize_source),
        dataframe["class_name"]
    )

    table = table.reindex(
        columns=CLASS_NAMES,
        fill_value=0
    )

    table["Total"] = table.sum(axis=1)

    table.insert(
        0,
        "Dataset",
        dataset_name
    )

    return table

if source_column is not None:

    if source_column in val_df.columns:

        val_composition = composition_table(
            val_df,
            "Validation"
        )

    else:

        val_composition = pd.DataFrame()

    test_composition = composition_table(
        test_df,
        "Test"
    )

    if not val_composition.empty:

        composition_df = pd.concat(
            [
                val_composition,
                test_composition
            ],
            ignore_index=True
        )

        print(composition_df.to_string(index=False))

        composition_path = os.path.join(
            OUTPUT_DIR,
            "exp4_validation_test_composition.csv"
        )

        composition_df.to_csv(
            composition_path,
            index=False
        )

print()
print("PATIENT GROUP ANALYSIS")

if patient_column is not None:

    patient_results = []

    for patient_group in sorted(
        pred_df["analysis_patient_group"].unique()
    ):

        patient_data = pred_df[
            pred_df["analysis_patient_group"] ==
            patient_group
        ]

        y_true = patient_data["true_label"].astype(str)
        y_pred = patient_data["predicted_label"].astype(str)

        patient_accuracy = accuracy_score(
            y_true,
            y_pred
        )

        patient_results.append({
            "patient_group": patient_group,
            "source": patient_data["analysis_source"].iloc[0]
            if source_column is not None
            else "Unknown",
            "samples": len(patient_data),
            "accuracy": patient_accuracy,
            "errors": int(
                (y_true != y_pred).sum()
            )
        })

    patient_results_df = pd.DataFrame(
        patient_results
    )

    patient_results_df = patient_results_df.sort_values(
        [
            "errors",
            "samples"
        ],
        ascending=[
            False,
            False
        ]
    )

    print(
        patient_results_df.head(20).to_string(
            index=False
        )
    )

    patient_results_path = os.path.join(
        OUTPUT_DIR,
        "exp4_patient_group_performance.csv"
    )

    patient_results_df.to_csv(
        patient_results_path,
        index=False
    )

    print()
    print("PATIENT GROUP SUMMARY")

    print(
        f"Total patient groups: "
        f"{patient_results_df['patient_group'].nunique()}"
    )

    print(
        f"Patient groups with errors: "
        f"{(patient_results_df['errors'] > 0).sum()}"
    )

    print(
        f"Patient groups with zero errors: "
        f"{(patient_results_df['errors'] == 0).sum()}"
    )

print()
print("FOCUSED COPD TO PNEUMONIA ANALYSIS")

copd_to_pneumonia = pred_df[
    (pred_df["true_label"] == "COPD") &
    (pred_df["predicted_label"] == "Pneumonia")
]

pneumonia_to_copd = pred_df[
    (pred_df["true_label"] == "Pneumonia") &
    (pred_df["predicted_label"] == "COPD")
]

print(
    f"COPD predicted as Pneumonia: "
    f"{len(copd_to_pneumonia)}"
)

print(
    f"Pneumonia predicted as COPD: "
    f"{len(pneumonia_to_copd)}"
)

if source_column is not None:

    print()
    print("COPD TO PNEUMONIA BY SOURCE")

    if len(copd_to_pneumonia) > 0:

        print(
            copd_to_pneumonia[
                "analysis_source"
            ].value_counts().to_string()
        )

    print()
    print("PNEUMONIA TO COPD BY SOURCE")

    if len(pneumonia_to_copd) > 0:

        print(
            pneumonia_to_copd[
                "analysis_source"
            ].value_counts().to_string()
        )

if patient_column is not None:

    print()
    print("COPD TO PNEUMONIA BY PATIENT GROUP")

    if len(copd_to_pneumonia) > 0:

        patient_error_counts = (
            copd_to_pneumonia[
                "analysis_patient_group"
            ]
            .value_counts()
        )

        print(
            patient_error_counts.to_string()
        )

print()
print("ERROR ANALYSIS FILES SAVED")

for filename in [
    "exp4_error_analysis_overall_confusion.csv",
    "exp4_error_pairs.csv",
    "exp4_error_analysis_class_metrics.csv",
    "exp4_test_source_class_distribution.csv",
    "exp4_source_level_performance.csv",
    "exp4_source_error_pairs.csv",
    "exp4_validation_test_composition.csv",
    "exp4_patient_group_performance.csv"
]:

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):
        print(path)

print()
print("AST Experiment 4 error analysis completed.")