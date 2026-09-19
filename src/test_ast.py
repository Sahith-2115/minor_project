import numpy as np
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

print("Loading AST feature extractor...")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

print("Loading AST model...")
model = ASTForAudioClassification.from_pretrained(MODEL_NAME)
model = model.cuda()
model.eval()

print("Creating 4 second test audio...")
audio = np.random.randn(64000).astype(np.float32)

print("Processing audio...")
inputs = extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

print("Input shape:", inputs["input_values"].shape)

inputs = {
    key: value.cuda()
    for key, value in inputs.items()
}

print("Running GPU forward pass...")

with torch.no_grad():
    outputs = model(**inputs)

print("Model loaded successfully")
print("Output shape:", outputs.logits.shape)
print("Device:", outputs.logits.device)