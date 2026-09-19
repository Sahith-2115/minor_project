from pathlib import Path
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "unified_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ast_exp3"

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
DURATION = 4
NUM_SAMPLES = SAMPLE_RATE * DURATION

CLASSES = ["Asthma", "Bronchial", "COPD", "Healthy", "Pneumonia"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 15
LR = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.1
PATIENCE = 4
MAX_GRAD_NORM = 1.0
SAMPLES_PER_CLASS = 300

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 80)
print("AST EXPERIMENT 3")
print("=" * 80)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LR}")
print(f"Samples per class per epoch: {SAMPLES_PER_CLASS}")
print("Balanced sampler: enabled")
print("Waveform augmentation: enabled")
print("SpecAugment: enabled")
print("Trainable AST layers: 8 to 11")
print("Frozen AST layers: 0 to 7")
print("=" * 80)


class ASTDataset(Dataset):
    def __init__(self, metadata_path, feature_extractor, training=False):
        self.df = pd.read_csv(metadata_path).reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.training = training

    def __len__(self):
        return len(self.df)

    def augment_waveform(self, y):
        if np.random.rand() < 0.7:
            gain = np.random.uniform(0.75, 1.25)
            y = y * gain

        if np.random.rand() < 0.5:
            noise_level = np.random.uniform(0.001, 0.008)
            noise = np.random.normal(0, noise_level, size=len(y))
            y = y + noise.astype(np.float32)

        if np.random.rand() < 0.5:
            max_shift = int(0.10 * SAMPLE_RATE)
            shift = np.random.randint(-max_shift, max_shift + 1)

            if shift > 0:
                y = np.pad(y, (shift, 0), mode="constant")[:len(y)]
            elif shift < 0:
                shift = abs(shift)
                y = np.pad(y, (0, shift), mode="constant")[shift:shift + len(y)]

        y = np.clip(y, -1.0, 1.0)

        return y

    def spec_augment(self, input_values):
        x = input_values.copy()

        if np.random.rand() < 0.5:
            time_mask_width = np.random.randint(10, 41)
            start = np.random.randint(0, max(1, x.shape[0] - time_mask_width))
            x[start:start + time_mask_width, :] = 0.0

        if np.random.rand() < 0.5:
            freq_mask_width = np.random.randint(4, 17)
            start = np.random.randint(0, max(1, x.shape[1] - freq_mask_width))
            x[:, start:start + freq_mask_width] = 0.0

        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = row["audio_path"]
        label = CLASS_TO_ID[row["class_name"]]

        y, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(y) > NUM_SAMPLES:
            if self.training:
                max_start = len(y) - NUM_SAMPLES
                start = np.random.randint(0, max_start + 1)
            else:
                start = (len(y) - NUM_SAMPLES) // 2

            y = y[start:start + NUM_SAMPLES]

        elif len(y) < NUM_SAMPLES:
            pad = NUM_SAMPLES - len(y)

            if self.training:
                left = np.random.randint(0, pad + 1)
            else:
                left = pad // 2

            right = pad - left

            y = np.pad(
                y,
                (left, right),
                mode="constant"
            )

        y = y.astype(np.float32)

        if self.training:
            y = self.augment_waveform(y)

        features = self.feature_extractor(
            y,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np"
        )

        input_values = features["input_values"][0]

        if self.training:
            input_values = self.spec_augment(input_values)

        return {
            "input_values": torch.tensor(
                input_values,
                dtype=torch.float32
            ),
            "labels": torch.tensor(
                label,
                dtype=torch.long
            )
        }


def create_balanced_sampler(dataset):
    labels = dataset.df["class_name"].map(CLASS_TO_ID).values

    class_counts = np.bincount(
        labels,
        minlength=len(CLASSES)
    )

    print()
    print("Original training class counts:")

    for i, class_name in enumerate(CLASSES):
        print(
            f"{class_name}: "
            f"{class_counts[i]}"
        )

    weights = np.zeros(len(labels), dtype=np.float64)

    for class_id in range(len(CLASSES)):
        indices = np.where(labels == class_id)[0]

        if len(indices) == 0:
            continue

        weights[indices] = 1.0 / len(indices)

    weights = torch.DoubleTensor(weights)

    num_samples = SAMPLES_PER_CLASS * len(CLASSES)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )

    return sampler


def freeze_ast_layers(model):
    ast = model.audio_spectrogram_transformer

    for parameter in ast.parameters():
        parameter.requires_grad = False

    for layer_index in range(8, 12):
        for parameter in ast.layers[layer_index].parameters():
            parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def evaluate(model, loader, split_name):
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"Evaluating {split_name}"
        ):
            input_values = batch["input_values"].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch["labels"].to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
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

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0
    )

    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(len(CLASSES))),
            average=None,
            zero_division=0
        )
    )

    print()
    print("=" * 80)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 80)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")

    print()
    print("Per class:")

    for i, class_name in enumerate(CLASSES):
        print(
            f"{class_name:12s} "
            f"P={per_class_precision[i]:.4f} "
            f"R={per_class_recall[i]:.4f} "
            f"F1={per_class_f1[i]:.4f} "
            f"Support={support[i]}"
        )

    print()
    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            digits=4,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES)))
    )

    print("Confusion Matrix:")
    print(cm)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "labels": all_labels,
        "predictions": all_predictions,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "support": support,
        "confusion_matrix": cm
    }


feature_extractor = ASTFeatureExtractor.from_pretrained(
    MODEL_NAME
)

train_dataset = ASTDataset(
    DATA_DIR / "train_metadata.csv",
    feature_extractor,
    training=True
)

val_dataset = ASTDataset(
    DATA_DIR / "validation_metadata.csv",
    feature_extractor,
    training=False
)

test_dataset = ASTDataset(
    DATA_DIR / "test_metadata.csv",
    feature_extractor,
    training=False
)

sampler = create_balanced_sampler(
    train_dataset
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print()
print(f"Train dataset: {len(train_dataset)}")
print(f"Validation dataset: {len(val_dataset)}")
print(f"Test dataset: {len(test_dataset)}")
print(f"Samples per epoch: {SAMPLES_PER_CLASS * len(CLASSES)}")

model = ASTForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(CLASSES),
    ignore_mismatched_sizes=True
)

model.config.id2label = {
    i: class_name
    for i, class_name in enumerate(CLASSES)
}

model.config.label2id = {
    class_name: i
    for i, class_name in enumerate(CLASSES)
}

freeze_ast_layers(model)

model.to(DEVICE)

total_parameters, trainable_parameters = count_parameters(
    model
)

print()
print("=" * 80)
print("PARAMETERS")
print("=" * 80)
print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")
print(
    f"Trainable percentage: "
    f"{100 * trainable_parameters / total_parameters:.2f}%"
)

print()
print("Trainable components:")

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name)

optimizer = torch.optim.AdamW(
    [
        {
            "params": [
                p for p in model.parameters()
                if p.requires_grad
            ],
            "lr": LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)

total_training_steps = (
    NUM_EPOCHS * len(train_loader)
)

warmup_steps = int(
    total_training_steps * WARMUP_RATIO
)


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step + 1) / max(
            1,
            warmup_steps
        )

    progress = (
        current_step - warmup_steps
    ) / max(
        1,
        total_training_steps - warmup_steps
    )

    return max(
        0.0,
        0.5 * (
            1.0 + np.cos(
                np.pi * progress
            )
        )
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

history = []

best_model_path = (
    MODEL_DIR /
    "best_ast_exp3_model.pt"
)

print()
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)

for epoch in range(
    1,
    NUM_EPOCHS + 1
):
    model.train()

    running_loss = 0.0
    sample_count = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    for batch in progress_bar:
        optimizer.zero_grad(
            set_to_none=True
        )

        input_values = batch["input_values"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):
            outputs = model(
                input_values=input_values,
                labels=labels
            )

            loss = outputs.loss

        old_scale = scaler.get_scale()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            [
                p for p in model.parameters()
                if p.requires_grad
            ],
            MAX_GRAD_NORM
        )

        scaler.step(optimizer)
        scaler.update()

        new_scale = scaler.get_scale()

        if new_scale >= old_scale:
            scheduler.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        sample_count += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    train_loss = running_loss / sample_count

    val_results = evaluate(
        model,
        val_loader,
        "validation"
    )

    val_f1 = val_results["macro_f1"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_results["loss"],
        "val_accuracy": val_results["accuracy"],
        "val_macro_precision": val_results["macro_precision"],
        "val_macro_recall": val_results["macro_recall"],
        "val_macro_f1": val_results["macro_f1"]
    })

    print()
    print(
        f"Epoch {epoch}: "
        f"Train Loss={train_loss:.4f} "
        f"Val Loss={val_results['loss']:.4f} "
        f"Val Acc={val_results['accuracy']:.4f} "
        f"Val Macro F1={val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "classes": CLASSES,
                "model_name": MODEL_NAME
            },
            best_model_path
        )

        print(
            f"New best model saved. "
            f"Validation Macro F1={val_f1:.4f}"
        )

    else:
        epochs_without_improvement += 1

        print(
            f"No improvement. "
            f"Patience: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

        if epochs_without_improvement >= PATIENCE:
            print()
            print("Early stopping triggered.")
            break

history_df = pd.DataFrame(history)

history_path = (
    REPORT_DIR /
    "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print()
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Best epoch: {best_epoch}")
print(f"Best validation Macro F1: {best_val_f1:.4f}")
print(f"Best model: {best_model_path}")

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print()
print("=" * 80)
print("FINAL VALIDATION EVALUATION")
print("=" * 80)

final_val_results = evaluate(
    model,
    val_loader,
    "validation"
)

print()
print("=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

final_test_results = evaluate(
    model,
    test_loader,
    "test"
)

test_report = classification_report(
    final_test_results["labels"],
    final_test_results["predictions"],
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

with open(
    REPORT_DIR / "test_classification_report.txt",
    "w"
) as f:
    f.write(test_report)

np.savetxt(
    REPORT_DIR / "test_confusion_matrix.csv",
    final_test_results["confusion_matrix"],
    fmt="%d",
    delimiter=","
)

predictions_df = pd.DataFrame({
    "true_label": [
        CLASSES[i]
        for i in final_test_results["labels"]
    ],
    "predicted_label": [
        CLASSES[i]
        for i in final_test_results["predictions"]
    ]
})

predictions_df.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)

metrics_df = pd.DataFrame({
    "class": CLASSES,
    "precision": final_test_results["per_class_precision"],
    "recall": final_test_results["per_class_recall"],
    "f1": final_test_results["per_class_f1"],
    "support": final_test_results["support"]
})

metrics_df.to_csv(
    REPORT_DIR / "test_per_class_metrics.csv",
    index=False
)

print()
print("=" * 80)
print("EXPERIMENT 3 COMPLETE")
print("=" * 80)
print(f"Best epoch: {best_epoch}")
print(f"Best validation Macro F1: {best_val_f1:.4f}")
print(
    f"Test Accuracy: "
    f"{final_test_results['accuracy']:.4f}"
)
print(
    f"Test Macro F1: "
    f"{final_test_results['macro_f1']:.4f}"
)
print(
    f"Test Macro Precision: "
    f"{final_test_results['macro_precision']:.4f}"
)
print(
    f"Test Macro Recall: "
    f"{final_test_results['macro_recall']:.4f}"
)
print()
print(f"Model saved to: {best_model_path}")
print(f"Reports saved to: {REPORT_DIR}")