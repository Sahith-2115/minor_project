import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

SEED = 42
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = 5
SAMPLE_RATE = 16000
AUDIO_DURATION = 4
MAX_AUDIO_SAMPLES = SAMPLE_RATE * AUDIO_DURATION
BATCH_SIZE = 4
NUM_WORKERS = 0

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"
TEST_METADATA = os.path.join(PROJECT_ROOT, "data/processed/unified_dataset/test_metadata.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs/ast_exp4/models/best_ast_exp4_model.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/ast_exp4/reports")

CLASS_NAMES = [
    "Asthma",
    "Bronchitis",
    "COPD",
    "Healthy",
    "Pneumonia"
]

CLASS_TO_ID = {
    "Asthma": 0,
    "Bronchitis": 1,
    "COPD": 2,
    "Healthy": 3,
    "Pneumonia": 4
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("AST EXPERIMENT 4 FINAL TEST EVALUATION")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

if not os.path.exists(TEST_METADATA):
    raise FileNotFoundError(f"Test metadata not found: {TEST_METADATA}")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Test metadata: {TEST_METADATA}")

print("Loading test metadata...")

test_df = pd.read_csv(TEST_METADATA)

required_columns = ["audio_path", "class_name"]

missing_columns = [
    column for column in required_columns
    if column not in test_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in test metadata: {missing_columns}"
    )

test_df["label"] = test_df["class_name"].map(CLASS_TO_ID)

if test_df["label"].isna().any():
    invalid_classes = test_df.loc[
        test_df["label"].isna(),
        "class_name"
    ].unique().tolist()
    raise ValueError(f"Unknown classes found: {invalid_classes}")

test_df["label"] = test_df["label"].astype(int)

print(f"Test samples: {len(test_df)}")

print("Test class distribution:")

for class_name in CLASS_NAMES:
    count = int((test_df["class_name"] == class_name).sum())
    print(f"{class_name}: {count}")

class RespiratoryAudioDataset(Dataset):
    def __init__(self, dataframe, feature_extractor):
        self.dataframe = dataframe.reset_index(drop=True)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataframe)

    def _load_audio(self, path):
        audio, sr = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) > MAX_AUDIO_SAMPLES:
            audio = audio[:MAX_AUDIO_SAMPLES]

        elif len(audio) < MAX_AUDIO_SAMPLES:
            audio = np.pad(
                audio,
                (0, MAX_AUDIO_SAMPLES - len(audio)),
                mode="constant"
            )

        return audio.astype(np.float32)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        audio_path = row["audio_path"]
        label = int(row["label"])

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        audio = self._load_audio(audio_path)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = inputs["input_values"].squeeze(0)

        return {
            "input_values": input_values,
            "label": torch.tensor(label, dtype=torch.long)
        }

print("Loading AST feature extractor...")

feature_extractor = ASTFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Creating test dataset...")

test_dataset = RespiratoryAudioDataset(
    test_df,
    feature_extractor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("Loading AST model...")

model = ASTForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label={
        index: class_name
        for index, class_name in enumerate(CLASS_NAMES)
    },
    label2id=CLASS_TO_ID,
    ignore_mismatched_sizes=True
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Best Experiment 4 checkpoint loaded.")
print("Running final test evaluation...")

all_predictions = []
all_labels = []
total_loss = 0.0
total_samples = 0

criterion = torch.nn.CrossEntropyLoss()

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "tqdm is required. Install it in the WSL venv terminal with: pip install tqdm"
    )

with torch.no_grad():
    progress_bar = tqdm(
        test_loader,
        desc="Testing",
        unit="batch"
    )

    for batch in progress_bar:
        input_values = batch["input_values"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["label"].to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda"
        ):
            outputs = model(
                input_values=input_values
            )

            loss = criterion(
                outputs.logits,
                labels
            )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

test_loss = total_loss / total_samples

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print()
print("FINAL TEST RESULTS")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print()
print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(cm)

metrics_path = os.path.join(
    OUTPUT_DIR,
    "final_test_metrics.csv"
)

metrics_df = pd.DataFrame({
    "metric": [
        "test_loss",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1"
    ],
    "value": [
        test_loss,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1
    ]
})

metrics_df.to_csv(
    metrics_path,
    index=False
)

report_path = os.path.join(
    OUTPUT_DIR,
    "final_classification_report.txt"
)

with open(report_path, "w") as file:
    file.write("AST EXPERIMENT 4 FINAL TEST EVALUATION\n\n")
    file.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
    file.write(f"Test samples: {len(test_df)}\n\n")
    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro Precision: {macro_precision:.4f}\n")
    file.write(f"Macro Recall: {macro_recall:.4f}\n")
    file.write(f"Macro F1: {macro_f1:.4f}\n")
    file.write(f"Weighted Precision: {weighted_precision:.4f}\n")
    file.write(f"Weighted Recall: {weighted_recall:.4f}\n")
    file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(np.array2string(cm))

cm_path = os.path.join(
    OUTPUT_DIR,
    "final_confusion_matrix.csv"
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_df.to_csv(cm_path)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

predictions_df = test_df.copy()

predictions_df["true_label"] = [
    CLASS_NAMES[index]
    for index in all_labels
]

predictions_df["predicted_label"] = [
    CLASS_NAMES[index]
    for index in all_predictions
]

predictions_df["correct"] = (
    predictions_df["true_label"] ==
    predictions_df["predicted_label"]
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print()
print(f"Metrics saved: {metrics_path}")
print(f"Classification report saved: {report_path}")
print(f"Confusion matrix saved: {cm_path}")
print(f"Predictions saved: {predictions_path}")
print("Experiment 4 final evaluation completed.")