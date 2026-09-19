import os
import pandas as pd

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "unified_dataset",
    "unified_dataset_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("==============================================")
    print("UNIFIED SPLIT SOURCE AUDIT")
    print("==============================================")

    print()
    print("Metadata:")
    print(METADATA_PATH)

    df = pd.read_csv(METADATA_PATH)

    print()
    print(
        f"Total samples: {len(df)}"
    )

    print()
    print("==============================================")
    print("SOURCE BY SPLIT")
    print("==============================================")

    source_summary = (
        df.groupby(
            ["split", "source"]
        )
        .agg(
            samples=("sample_id", "count"),
            patient_groups=("patient_group", "nunique")
        )
        .reset_index()
    )

    print(
        source_summary.to_string(index=False)
    )

    print()
    print("==============================================")
    print("SOURCE BY SPLIT AND CLASS")
    print("==============================================")

    class_summary = (
        df.groupby(
            ["split", "class_name", "source"]
        )
        .agg(
            samples=("sample_id", "count"),
            patient_groups=("patient_group", "nunique")
        )
        .reset_index()
    )

    print(
        class_summary.to_string(index=False)
    )

    print()
    print("==============================================")
    print("CLASS PERCENTAGE WITHIN EACH SPLIT")
    print("==============================================")

    class_counts = (
        df.groupby(
            ["split", "class_name"]
        )
        .size()
        .reset_index(name="samples")
    )

    split_totals = (
        df.groupby("split")
        .size()
        .reset_index(name="split_total")
    )

    class_counts = class_counts.merge(
        split_totals,
        on="split"
    )

    class_counts["percentage"] = (
        100
        * class_counts["samples"]
        / class_counts["split_total"]
    ).round(2)

    print(
        class_counts.to_string(index=False)
    )

    source_path = os.path.join(
        OUTPUT_DIR,
        "unified_split_source_summary.csv"
    )

    class_path = os.path.join(
        OUTPUT_DIR,
        "unified_split_source_class_summary.csv"
    )

    percentage_path = os.path.join(
        OUTPUT_DIR,
        "unified_split_class_percentages.csv"
    )

    source_summary.to_csv(
        source_path,
        index=False
    )

    class_summary.to_csv(
        class_path,
        index=False
    )

    class_counts.to_csv(
        percentage_path,
        index=False
    )

    print()
    print("==============================================")
    print("REPORTS SAVED")
    print("==============================================")

    print(source_path)
    print(class_path)
    print(percentage_path)

    print()
    print("==============================================")
    print("AUDIT COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()