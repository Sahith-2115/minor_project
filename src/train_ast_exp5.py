from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import librosa
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

SEED = 42

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")

TRAIN_METADATA = PROJECT_ROOT / "data/processed/unified_dataset/train_metadata.csv"
VAL_METADATA = PROJECT_ROOT / "data/processed/unified_dataset/validation_metadata.csv"
TEST_METADATA = PROJECT_ROOT / "data/processed/unified_dataset/test_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs/ast_exp5"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

SAMPLE_RATE = 16000
DURATION = 4
NUM_SAMPLES = SAMPLE_RATE * DURATION

CLASSES = [
    "Asthma",
    "Bronchitis",
    "COPD",
    "Healthy",
    "Pneumonia"
]

LABEL2ID = {
    name: i
    for i, name in enumerate(CLASSES)
}

ID2LABEL = {
    i: name
    for name, i in LABEL2ID.items()
}

BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 15

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.10

PATIENCE = 4
MAX_GRAD_NORM = 1.0

SAMPLES_PER_CLASS_PER_EPOCH = 300

AUGMENTATION = True

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RespiratoryASTDataset(Dataset):
    def __init__(
        self,
        metadata,
        feature_extractor,
        training=False
    ):
        self.metadata = metadata.reset_index(
            drop=True
        )
        self.feature_extractor = feature_extractor
        self.training = training

    def __len__(self):
        return len(self.metadata)

    def load_audio(self, path):
        audio, _ = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) > NUM_SAMPLES:
            audio = audio[:NUM_SAMPLES]

        elif len(audio) < NUM_SAMPLES:
            pad_length = NUM_SAMPLES - len(audio)

            audio = np.pad(
                audio,
                (0, pad_length),
                mode="constant"
            )

        return audio.astype(
            np.float32
        )

    def augment_audio(self, audio):
        if not self.training:
            return audio

        if random.random() < 0.5:
            gain = random.uniform(
                0.85,
                1.15
            )
            audio = audio * gain

        if random.random() < 0.25:
            noise_level = random.uniform(
                0.0005,
                0.003
            )

            noise = np.random.normal(
                0,
                noise_level,
                size=audio.shape
            ).astype(
                np.float32
            )

            audio = audio + noise

        if random.random() < 0.25:
            max_shift = int(
                0.05 * SAMPLE_RATE
            )

            shift = random.randint(
                -max_shift,
                max_shift
            )

            if shift > 0:
                audio = np.pad(
                    audio[:-shift],
                    (shift, 0),
                    mode="constant"
                )

            elif shift < 0:
                shift = abs(shift)

                audio = np.pad(
                    audio[shift:],
                    (0, shift),
                    mode="constant"
                )

        audio = np.clip(
            audio,
            -1.0,
            1.0
        )

        return audio.astype(
            np.float32
        )

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        audio = self.load_audio(
            row["audio_path"]
        )

        audio = self.augment_audio(
            audio
        )

        features = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = features[
            "input_values"
        ].squeeze(0)

        label = LABEL2ID[
            row["class_name"]
        ]

        return {
            "input_values": input_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long
            )
        }

def create_patient_balanced_sampler(metadata):
    labels = metadata[
        "class_name"
    ].map(
        LABEL2ID
    ).values

    patient_groups = metadata[
        "patient_group"
    ].astype(str).values

    class_counts = np.bincount(
        labels,
        minlength=len(CLASSES)
    )

    print("\nTraining class counts:")

    for i, class_name in enumerate(CLASSES):
        print(
            f"{class_name}: "
            f"{class_counts[i]}"
        )

    class_patient_groups = {}

    for class_id, class_name in enumerate(CLASSES):
        mask = labels == class_id

        groups = sorted(
            set(
                patient_groups[mask]
            )
        )

        class_patient_groups[class_id] = groups

        print(
            f"{class_name} patients: "
            f"{len(groups)}"
        )

    sample_weights = np.zeros(
        len(metadata),
        dtype=np.float64
    )

    for i in range(len(metadata)):
        class_id = labels[i]
        patient_group = patient_groups[i]

        num_patients = len(
            class_patient_groups[class_id]
        )

        if num_patients == 0:
            sample_weights[i] = 0.0
            continue

        patient_count = np.sum(
            (labels == class_id) &
            (patient_groups == patient_group)
        )

        if patient_count == 0:
            sample_weights[i] = 0.0
            continue

        sample_weights[i] = (
            1.0 /
            (
                num_patients *
                patient_count
            )
        )

    samples_per_epoch = (
        SAMPLES_PER_CLASS_PER_EPOCH *
        len(CLASSES)
    )

    sampler = WeightedRandomSampler(
        weights=torch.tensor(
            sample_weights,
            dtype=torch.double
        ),
        num_samples=samples_per_epoch,
        replacement=True
    )

    print(
        "\nPatient balanced samples per epoch: "
        f"{samples_per_epoch}"
    )

    print(
        "Target samples per class per epoch: "
        f"{SAMPLES_PER_CLASS_PER_EPOCH}"
    )

    print(
        "Sampling weight rule: "
        "class balance × patient balance"
    )

    return sampler

def evaluate(
    model,
    loader,
    split_name
):
    model.eval()

    all_labels = []
    all_predictions = []
    losses = []

    with torch.no_grad():
        for batch in loader:
            input_values = batch[
                "input_values"
            ].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch[
                "labels"
            ].to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                input_values=input_values,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(
                logits,
                dim=1
            )

            losses.append(
                loss.item()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(
                range(len(CLASSES))
            ),
            zero_division=0
        )
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(CLASSES))
        ),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(CLASSES))
        )
    )

    print(
        f"\n{split_name} Results"
    )

    print(
        f"Loss: {np.mean(losses):.4f}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(
                range(len(CLASSES))
            ),
            target_names=CLASSES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(cm)

    metrics = {
        "loss": float(
            np.mean(losses)
        ),
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    return metrics, y_true, y_pred

def save_predictions(
    metadata,
    y_true,
    y_pred,
    split_name
):
    result = metadata.copy()

    result["true_label"] = [
        ID2LABEL[int(x)]
        for x in y_true
    ]

    result["predicted_label"] = [
        ID2LABEL[int(x)]
        for x in y_pred
    ]

    result["correct"] = (
        result["true_label"]
        ==
        result["predicted_label"]
    )

    path = (
        REPORT_DIR
        / f"{split_name}_predictions.csv"
    )

    result.to_csv(
        path,
        index=False
    )

    print(
        f"Predictions saved: {path}"
    )

def save_metrics(
    metrics,
    split_name
):
    path = (
        REPORT_DIR
        / f"{split_name}_metrics.json"
    )

    with open(
        path,
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    print(
        f"Metrics saved: {path}"
    )

def get_source_metrics(
    metadata,
    y_true,
    y_pred
):
    result = metadata.copy()

    result["true_id"] = y_true
    result["pred_id"] = y_pred

    rows = []

    for source in sorted(
        result["source"].unique()
    ):
        source_df = result[
            result["source"] == source
        ]

        y_source_true = (
            source_df["true_id"].values
        )

        y_source_pred = (
            source_df["pred_id"].values
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_source_true,
                y_source_pred,
                labels=list(
                    range(len(CLASSES))
                ),
                zero_division=0
            )
        )

        for i, class_name in enumerate(
            CLASSES
        ):
            rows.append({
                "source": source,
                "class_name": class_name,
                "precision": float(
                    precision[i]
                ),
                "recall": float(
                    recall[i]
                ),
                "f1": float(
                    f1[i]
                ),
                "support": int(
                    support[i]
                )
            })

    source_metrics = pd.DataFrame(
        rows
    )

    return source_metrics

def get_patient_sampling_summary(metadata):
    rows = []

    for class_name in CLASSES:
        class_df = metadata[
            metadata["class_name"] == class_name
        ]

        patient_counts = (
            class_df
            .groupby("patient_group")
            .size()
        )

        if len(patient_counts) == 0:
            continue

        rows.append({
            "class_name": class_name,
            "patient_count": int(
                patient_counts.shape[0]
            ),
            "min_samples_per_patient": int(
                patient_counts.min()
            ),
            "max_samples_per_patient": int(
                patient_counts.max()
            ),
            "mean_samples_per_patient": float(
                patient_counts.mean()
            ),
            "median_samples_per_patient": float(
                patient_counts.median()
            )
        })

    return pd.DataFrame(rows)

def main():
    seed_everything(SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nAST EXPERIMENT 5"
    )

    print(
        "PATIENT BALANCED SAMPLING"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Samples per class per epoch: "
        f"{SAMPLES_PER_CLASS_PER_EPOCH}"
    )

    print(
        f"Waveform augmentation: {AUGMENTATION}"
    )

    print(
        "\nLoading metadata..."
    )

    train_df = pd.read_csv(
        TRAIN_METADATA
    )

    val_df = pd.read_csv(
        VAL_METADATA
    )

    test_df = pd.read_csv(
        TEST_METADATA
    )

    print(
        f"Train samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    for class_name in CLASSES:
        if class_name not in train_df[
            "class_name"
        ].unique():
            raise ValueError(
                f"Training data missing class: "
                f"{class_name}"
            )

        if class_name not in val_df[
            "class_name"
        ].unique():
            raise ValueError(
                f"Validation data missing class: "
                f"{class_name}"
            )

        if class_name not in test_df[
            "class_name"
        ].unique():
            raise ValueError(
                f"Test data missing class: "
                f"{class_name}"
            )

    print(
        "\nTraining patient distribution:"
    )

    patient_summary = get_patient_sampling_summary(
        train_df
    )

    print(
        patient_summary.to_string(
            index=False
        )
    )

    patient_summary.to_csv(
        REPORT_DIR /
        "training_patient_distribution.csv",
        index=False
    )

    print(
        "\nLoading AST feature extractor..."
    )

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "\nCreating datasets..."
    )

    train_dataset = RespiratoryASTDataset(
        train_df,
        feature_extractor,
        training=AUGMENTATION
    )

    val_dataset = RespiratoryASTDataset(
        val_df,
        feature_extractor,
        training=False
    )

    test_dataset = RespiratoryASTDataset(
        test_df,
        feature_extractor,
        training=False
    )

    sampler = create_patient_balanced_sampler(
        train_df
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

    print(
        "\nLoading pretrained AST model..."
    )

    model = (
        ASTForAudioClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(CLASSES),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True
        )
    )

    model.to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable percentage: "
        f"{100 * trainable_params / total_params:.2f}%"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_training_steps = (
        NUM_EPOCHS
        * len(train_loader)
    )

    warmup_steps = int(
        total_training_steps
        * WARMUP_RATIO
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                total_training_steps
                - warmup_steps
            )
        )
    )

    use_amp = torch.cuda.is_available()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    best_model_path = (
        MODEL_DIR
        / "best_ast_exp5_model.pt"
    )

    print(
        "\nStarting Experiment 5 training..."
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):
        model.train()

        running_loss = 0.0
        batches = 0

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        for batch in train_loader:
            input_values = batch[
                "input_values"
            ].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch[
                "labels"
            ].to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):
                outputs = model(
                    input_values=input_values,
                    labels=labels
                )

                loss = outputs.loss

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
            )

            old_scale = scaler.get_scale()

            scaler.step(
                optimizer
            )

            scaler.update()

            new_scale = scaler.get_scale()

            if new_scale >= old_scale:
                scheduler.step()

            running_loss += loss.item()
            batches += 1

        train_loss = (
            running_loss / batches
        )

        val_metrics, val_true, val_pred = evaluate(
            model,
            val_loader,
            "Validation"
        )

        val_f1 = val_metrics[
            "macro_f1"
        ]

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(
                train_loss
            ),
            "validation_loss": val_metrics[
                "loss"
            ],
            "validation_accuracy": val_metrics[
                "accuracy"
            ],
            "validation_macro_precision": val_metrics[
                "macro_precision"
            ],
            "validation_macro_recall": val_metrics[
                "macro_recall"
            ],
            "validation_macro_f1": val_metrics[
                "macro_f1"
            ],
            "validation_weighted_f1": val_metrics[
                "weighted_f1"
            ]
        }

        history.append(
            epoch_record
        )

        print(
            f"\nEpoch {epoch} Summary"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "model_name": MODEL_NAME,
                "epoch": epoch,
                "validation_macro_f1": best_val_f1
            }

            torch.save(
                checkpoint,
                best_model_path
            )

            print(
                "New best model saved."
            )

            print(
                f"Best validation Macro F1: "
                f"{best_val_f1:.4f}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement."
            )

            print(
                f"Patience: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                "\nEarly stopping triggered."
            )
            break

    history_path = (
        REPORT_DIR
        / "training_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False
    )

    print(
        f"\nTraining history saved: "
        f"{history_path}"
    )

    print(
        "\nBEST MODEL SUMMARY"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Best model: "
        f"{best_model_path}"
    )

    print(
        "\nLoading best checkpoint..."
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "\nFinal validation evaluation..."
    )

    final_val_metrics, final_val_true, final_val_pred = evaluate(
        model,
        val_loader,
        "Final Validation"
    )

    save_metrics(
        final_val_metrics,
        "final_validation"
    )

    save_predictions(
        val_df,
        final_val_true,
        final_val_pred,
        "validation"
    )

    print(
        "\nFinal test evaluation..."
    )

    test_metrics, test_true, test_pred = evaluate(
        model,
        test_loader,
        "Test"
    )

    save_metrics(
        test_metrics,
        "test"
    )

    save_predictions(
        test_df,
        test_true,
        test_pred,
        "test"
    )

    source_metrics = get_source_metrics(
        test_df,
        test_true,
        test_pred
    )

    source_metrics_path = (
        REPORT_DIR
        / "test_source_metrics.csv"
    )

    source_metrics.to_csv(
        source_metrics_path,
        index=False
    )

    print(
        f"\nSource wise test metrics saved: "
        f"{source_metrics_path}"
    )

    print(
        "\nEXPERIMENT 5 FINAL RESULTS"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Macro Precision: "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Test Macro Recall: "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test Weighted F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        "\nTest classification report:"
    )

    print(
        classification_report(
            test_true,
            test_pred,
            labels=list(
                range(len(CLASSES))
            ),
            target_names=CLASSES,
            zero_division=0
        )
    )

    print(
        "Test confusion matrix:"
    )

    print(
        confusion_matrix(
            test_true,
            test_pred,
            labels=list(
                range(len(CLASSES))
            )
        )
    )

    print(
        "\nExperiment 5 completed."
    )

if __name__ == "__main__":
    main()