import sys
sys.path.insert(0, "src")

import torch
from torch.utils.data import DataLoader
from ast_dataset import ASTAudioDataset
from transformers import AutoFeatureExtractor, ASTForAudioClassification

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

print("=" * 60)
print("AST GPU TRAINING TEST")
print("=" * 60)

print("Loading feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Loading dataset...")

dataset = ASTAudioDataset(
    "data/processed/unified_dataset/train_metadata.csv",
    feature_extractor=feature_extractor
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print("Loading one batch...")

batch = next(iter(loader))

print(
    "Batch shape:",
    tuple(batch["input_values"].shape)
)

print(
    "Labels:",
    batch["label"].tolist()
)

print("Loading AST model...")

model = ASTForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5,
    ignore_mismatched_sizes=True
)

model = model.cuda()
model.train()

inputs = batch["input_values"].cuda(
    non_blocking=True
)

labels = batch["label"].cuda(
    non_blocking=True
)

print("Running forward pass...")

with torch.amp.autocast(
    device_type="cuda"
):
    outputs = model(
        input_values=inputs,
        labels=labels
    )

loss = outputs.loss

print(
    "Logits shape:",
    tuple(outputs.logits.shape)
)

print(
    "Loss:",
    loss.item()
)

print("Running backward pass...")

loss.backward()

print("Backward pass: OK")

allocated = (
    torch.cuda.memory_allocated()
    / (1024 ** 3)
)

reserved = (
    torch.cuda.memory_reserved()
    / (1024 ** 3)
)

print(
    f"GPU memory allocated: {allocated:.2f} GB"
)

print(
    f"GPU memory reserved: {reserved:.2f} GB"
)

print("=" * 60)
print("AST GPU TRAINING TEST PASSED")
print("=" * 60)