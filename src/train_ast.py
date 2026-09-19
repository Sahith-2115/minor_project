import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import ASTForAudioClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from ast_dataset import create_dataset, CLASS_NAMES

SEED = 42
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
PROJECT_ROOT = "/home/sahith/projects/minor_project_408"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ast_exp2")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BATCH_SIZE = 4
NUM_EPOCHS = 15
LR = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.1
PATIENCE = 4
MAX_GRAD_NORM = 1.0
SAMPLES_PER_CLASS = 300
NUM_WORKERS = 0
USE_AMP = torch.cuda.is_available()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CappedBalancedSampler(Sampler):
    def __init__(self, labels, num_classes, samples_per_class, seed=42):
        self.labels = np.asarray(labels)
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0
        self.class_indices = {}
        for c in range(num_classes):
            indices = np.where(self.labels == c)[0]
            if len(indices) == 0:
                raise ValueError(f"Class {c} has no samples.")
            self.class_indices[c] = indices

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = []

        for c in range(self.num_classes):
            indices = self.class_indices[c]

            if len(indices) >= self.samples_per_class:
                chosen = rng.choice(
                    indices,
                    size=self.samples_per_class,
                    replace=False
                )
            else:
                chosen = rng.choice(
                    indices,
                    size=self.samples_per_class,
                    replace=True
                )

            selected.extend(chosen.tolist())

        rng.shuffle(selected)
        return iter(selected)

    def __len__(self):
        return self.num_classes * self.samples_per_class

def apply_spec_augment(x):
    x = x.clone()

    if random.random() < 0.5:
        freq_width = random.randint(4, 10)
        freq_start = random.randint(0, x.shape[1] - freq_width)
        x[:, freq_start:freq_start + freq_width] = 0.0

    if random.random() < 0.5:
        time_width = random.randint(20, 64)
        time_start = random.randint(0, x.shape[0] - time_width)
        x[time_start:time_start + time_width, :] = 0.0

    return x

class TrainDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        x = torch.as_tensor(item["input_values"], dtype=torch.float32)
        x = apply_spec_augment(x)

        return {
            "input_values": x,
            "labels": torch.as_tensor(item["label"], dtype=torch.long)
        }

class EvalDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        return {
            "input_values": torch.as_tensor(
                item["input_values"],
                dtype=torch.float32
            ),
            "labels": torch.as_tensor(
                item["label"],
                dtype=torch.long
            )
        }

def collate_fn(batch):
    input_values = torch.stack(
        [item["input_values"] for item in batch]
    )
    labels = torch.stack(
        [item["labels"] for item in batch]
    )

    return {
        "input_values": input_values,
        "labels": labels
    }

def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(
                device,
                non_blocking=True
            )
            labels = batch["labels"].to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                outputs = model(
                    input_values=input_values,
                    labels=labels
                )

            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": np.array(all_labels),
        "preds": np.array(all_preds)
    }

def print_metrics(metrics, prefix=""):
    print(
        f"{prefix}Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy'] * 100:.2f}% | "
        f"Macro Precision: {metrics['macro_precision']:.4f} | "
        f"Macro Recall: {metrics['macro_recall']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:<12} "
            f"P: {metrics['precision'][i]:.4f} "
            f"R: {metrics['recall'][i]:.4f} "
            f"F1: {metrics['f1'][i]:.4f}"
        )

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("AST EXPERIMENT 2 TEST EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Model: {MODEL_NAME}")

    val_base = create_dataset("validation")
    test_base = create_dataset("test")

    val_dataset = EvalDatasetWrapper(val_base)
    test_dataset = EvalDatasetWrapper(test_base)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    print()
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    label2id = {
        name: i for i, name in enumerate(CLASS_NAMES)
    }

    id2label = {
        i: name for i, name in enumerate(CLASS_NAMES)
    }

    print()
    print("Loading AST model...")

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    model.to(device)

    best_model_path = os.path.join(
        MODEL_DIR,
        "best_ast_exp2_model.pt"
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{best_model_path}"
        )

    print()
    print("Loading Experiment 2 checkpoint...")
    print(best_model_path)

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_epoch = checkpoint["epoch"]
    best_val_f1 = checkpoint["val_macro_f1"]

    print()
    print(f"Checkpoint epoch: {best_epoch}")
    print(f"Validation Macro F1: {best_val_f1:.4f}")

    print()
    print("Evaluating validation set...")

    val_metrics = evaluate(
        model,
        val_loader,
        device
    )

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print_metrics(
        val_metrics,
        prefix="Validation "
    )

    print()
    print("Evaluating test set...")

    test_metrics = evaluate(
        model,
        test_loader,
        device
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print_metrics(
        test_metrics,
        prefix="Test "
    )

    report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print()
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(
        test_metrics["labels"],
        test_metrics["preds"],
        labels=list(range(len(CLASS_NAMES)))
    )

    print("Confusion Matrix:")
    print(cm)

    report_path = os.path.join(
        REPORT_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    cm_path = os.path.join(
        REPORT_DIR,
        "confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(cm_path)

    predictions_path = os.path.join(
        REPORT_DIR,
        "test_predictions.csv"
    )

    pd.DataFrame({
        "true_label": [
            CLASS_NAMES[i]
            for i in test_metrics["labels"]
        ],
        "predicted_label": [
            CLASS_NAMES[i]
            for i in test_metrics["preds"]
        ]
    }).to_csv(
        predictions_path,
        index=False
    )

    metrics_path = os.path.join(
        REPORT_DIR,
        "final_metrics.txt"
    )

    with open(metrics_path, "w") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(
            f"Best validation Macro F1: "
            f"{best_val_f1:.6f}\n"
        )
        f.write(
            f"Test loss: "
            f"{test_metrics['loss']:.6f}\n"
        )
        f.write(
            f"Test accuracy: "
            f"{test_metrics['accuracy']:.6f}\n"
        )
        f.write(
            f"Test macro precision: "
            f"{test_metrics['macro_precision']:.6f}\n"
        )
        f.write(
            f"Test macro recall: "
            f"{test_metrics['macro_recall']:.6f}\n"
        )
        f.write(
            f"Test macro F1: "
            f"{test_metrics['macro_f1']:.6f}\n"
        )

    print()
    print("Results saved to:")
    print(REPORT_DIR)

    print()
    print("=" * 70)
    print("EXPERIMENT 2 TEST EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()