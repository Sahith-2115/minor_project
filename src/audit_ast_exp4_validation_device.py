import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import librosa
from tqdm import tqdm

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"

VALIDATION_METADATA = os.path.join(
    PROJECT_ROOT,
    "data/processed/unified_dataset/validation_metadata.csv"
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/models/best_ast_exp4_model.pt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/ast_exp4/reports/patient_134_audit/device_audit"
)

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

CLASS_NAMES = [
    "Asthma",
    "Bronchitis",
    "COPD",
    "Healthy",
    "Pneumonia"
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

class RespiratoryDataset(Dataset):
    def __init__(
        self,
        metadata,
        feature_extractor
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        path = row["audio_path"]

        try:
            audio, sr = librosa.load(
                path,
                sr=16000,
                mono=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load audio: {path}\n{e}"
            )

        target_length = 16000 * 4

        if len(audio) > target_length:
            audio = audio[:target_length]

        elif len(audio) < target_length:
            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )

        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs["input_values"].squeeze(0)

        return {
            "input_values": input_values,
            "label": CLASS_TO_ID[row["class_name"]],
            "sample_id": row["sample_id"],
            "audio_path": row["audio_path"],
            "patient_group": row["patient_group"],
            "device": extract_device(row["audio_path"]),
            "class_name": row["class_name"]
        }

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

def collate_fn(batch):
    return {
        "input_values": torch.stack(
            [item["input_values"] for item in batch]
        ),
        "label": torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long
        ),
        "sample_id": [
            item["sample_id"]
            for item in batch
        ],
        "audio_path": [
            item["audio_path"]
            for item in batch
        ],
        "patient_group": [
            item["patient_group"]
            for item in batch
        ],
        "device": [
            item["device"]
            for item in batch
        ],
        "class_name": [
            item["class_name"]
            for item in batch
        ]
    }

def load_model():
    print(
        f"Loading AST model from: {MODEL_NAME}"
    )

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        label2id=CLASS_TO_ID,
        id2label={
            index: name
            for index, name in enumerate(CLASS_NAMES)
        },
        ignore_mismatched_sizes=True
    )

    if not os.path.exists(CHECKPOINT):
        print(
            f"ERROR: Checkpoint not found: {CHECKPOINT}"
        )
        sys.exit(1)

    print(
        f"Loading Experiment 4 checkpoint: {CHECKPOINT}"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print(
        f"Missing keys: {len(missing)}"
    )

    print(
        f"Unexpected keys: {len(unexpected)}"
    )

    if len(missing) > 0:
        print("Missing keys:")
        for key in missing[:20]:
            print(key)

    if len(unexpected) > 0:
        print("Unexpected keys:")
        for key in unexpected[:20]:
            print(key)

    model = model.to(DEVICE)
    model.eval()

    return model

def evaluate(
    model,
    loader
):
    records = []

    total_loss = 0.0
    total_samples = 0

    loss_function = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Evaluating validation set",
            unit="batch"
        ):
            inputs = batch["input_values"].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch["label"].to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                input_values=inputs
            )

            logits = outputs.logits

            loss = loss_function(
                logits,
                labels
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            for index in range(batch_size):
                true_id = labels[index].item()
                pred_id = predictions[index].item()

                records.append({
                    "sample_id": batch["sample_id"][index],
                    "audio_path": batch["audio_path"][index],
                    "patient_group": batch["patient_group"][index],
                    "device": batch["device"][index],
                    "true_class": CLASS_NAMES[true_id],
                    "predicted_class": CLASS_NAMES[pred_id],
                    "correct": true_id == pred_id
                })

    results = pd.DataFrame(records)

    loss = total_loss / total_samples

    accuracy = results["correct"].mean()

    return results, loss, accuracy

def print_group_performance(
    results,
    group_column,
    title
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    rows = []

    for group, data in results.groupby(
        group_column
    ):
        rows.append({
            group_column: group,
            "samples": len(data),
            "correct": int(data["correct"].sum()),
            "accuracy": data["correct"].mean()
        })

    summary = pd.DataFrame(rows)

    if len(summary) > 0:
        summary = summary.sort_values(
            "samples",
            ascending=False
        )

        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    return summary

def print_confusion(
    results,
    title
):
    print()
    print(title)

    confusion = pd.crosstab(
        results["true_class"],
        results["predicted_class"]
    )

    confusion = confusion.reindex(
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
        fill_value=0
    )

    print(
        confusion.to_string()
    )

    return confusion

def main():
    print(
        f"Device: {DEVICE}"
    )

    if not os.path.exists(
        VALIDATION_METADATA
    ):
        print(
            f"ERROR: Validation metadata not found: "
            f"{VALIDATION_METADATA}"
        )
        sys.exit(1)

    metadata = pd.read_csv(
        VALIDATION_METADATA
    )

    print()
    print(
        f"Validation samples: {len(metadata)}"
    )

    feature_extractor = (
        ASTFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    model = load_model()

    dataset = RespiratoryDataset(
        metadata,
        feature_extractor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    results, loss, accuracy = evaluate(
        model,
        loader
    )

    print()
    print(
        f"Validation Loss: {loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{accuracy:.4f}"
    )

    print_confusion(
        results,
        "Overall Validation Confusion Matrix"
    )

    device_summary = print_group_performance(
        results,
        "device",
        "Validation Performance by Device"
    )

    patient_summary = print_group_performance(
        results,
        "patient_group",
        "Validation Performance by Patient"
    )

    print()
    print("=" * 70)
    print("LittC2SE VALIDATION ANALYSIS")
    print("=" * 70)

    litt = results[
        results["device"] == "LittC2SE"
    ].copy()

    if len(litt) == 0:
        print(
            "No LittC2SE samples found."
        )
    else:
        print()
        print(
            f"LittC2SE validation samples: "
            f"{len(litt)}"
        )

        print(
            f"LittC2SE correct: "
            f"{litt['correct'].sum()}"
        )

        print(
            f"LittC2SE accuracy: "
            f"{litt['correct'].mean():.4f}"
        )

        print()
        print("LittC2SE confusion matrix")

        print_confusion(
            litt,
            "LittC2SE"
        )

        print()
        print("LittC2SE performance by patient")

        litt_patient_summary = print_group_performance(
            litt,
            "patient_group",
            "LittC2SE COPD validation patients"
        )

        litt_patient_summary.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "littc2se_validation_patient_performance.csv"
            ),
            index=False
        )

        litt.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "littc2se_validation_predictions.csv"
            ),
            index=False
        )

    print()
    print("=" * 70)
    print("COPD PERFORMANCE BY DEVICE")
    print("=" * 70)

    copd = results[
        results["true_class"] == "COPD"
    ].copy()

    copd_device = print_group_performance(
        copd,
        "device",
        "Validation COPD Performance by Device"
    )

    copd_device.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_copd_performance_by_device.csv"
        ),
        index=False
    )

    print()
    print("=" * 70)
    print("COPD PERFORMANCE BY PATIENT")
    print("=" * 70)

    copd_patient = print_group_performance(
        copd,
        "patient_group",
        "Validation COPD Performance by Patient"
    )

    copd_patient.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_copd_performance_by_patient.csv"
        ),
        index=False
    )

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_predictions_exp4.csv"
        ),
        index=False
    )

    device_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_performance_by_device.csv"
        ),
        index=False
    )

    patient_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_performance_by_patient.csv"
        ),
        index=False
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