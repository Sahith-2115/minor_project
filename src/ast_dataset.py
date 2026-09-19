import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
DURATION = 4
NUM_SAMPLES = SAMPLE_RATE * DURATION

CLASS_NAMES = [
    "Asthma",
    "Bronchial",
    "COPD",
    "Healthy",
    "Pneumonia"
]

CLASS_TO_ID = {
    "Asthma": 0,
    "Bronchial": 1,
    "COPD": 2,
    "Healthy": 3,
    "Pneumonia": 4
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "unified_dataset"
)

class ASTAudioDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        feature_extractor=None
    ):
        self.metadata = pd.read_csv(metadata_path).reset_index(drop=True)

        required_columns = [
            "audio_path",
            "class_name"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.metadata.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        invalid_classes = sorted(
            set(self.metadata["class_name"]) - set(CLASS_NAMES)
        )

        if invalid_classes:
            raise ValueError(
                f"Unknown classes found: {invalid_classes}"
            )

        self.feature_extractor = feature_extractor

        if self.feature_extractor is None:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                MODEL_NAME
            )

    def __len__(self):
        return len(self.metadata)

    def _load_audio(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        audio, sr = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) > NUM_SAMPLES:
            audio = audio[:NUM_SAMPLES]

        elif len(audio) < NUM_SAMPLES:
            padding = NUM_SAMPLES - len(audio)
            audio = np.pad(
                audio,
                (0, padding),
                mode="constant"
            )

        audio = audio.astype(np.float32)

        return audio

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        audio_path = row["audio_path"]
        class_name = row["class_name"]

        audio = self._load_audio(audio_path)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = inputs["input_values"].squeeze(0)

        label = CLASS_TO_ID[class_name]

        return {
            "input_values": input_values,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
            "audio_path": audio_path
        }


def get_metadata_path(split):
    if split not in ["train", "validation", "test"]:
        raise ValueError(
            "split must be train, validation, or test"
        )

    return os.path.join(
        DATASET_DIR,
        f"{split}_metadata.csv"
    )


def create_dataset(split):
    metadata_path = get_metadata_path(split)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    return ASTAudioDataset(metadata_path)


if __name__ == "__main__":
    print("=" * 60)
    print("AST DATASET SANITY TEST")
    print("=" * 60)

    metadata_path = get_metadata_path("train")

    print(f"Metadata: {metadata_path}")

    dataset = ASTAudioDataset(metadata_path)

    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Class mapping: {CLASS_TO_ID}")

    print("\nLoading first sample...")

    sample = dataset[0]

    print(
        f"Input shape: {tuple(sample['input_values'].shape)}"
    )

    print(
        f"Label: {sample['label'].item()}"
    )

    print(
        f"Class: {sample['class_name']}"
    )

    print(
        f"Audio path: {sample['audio_path']}"
    )

    print("\nChecking all five classes...")

    class_found = {}

    for index in range(len(dataset)):
        class_name = dataset.metadata.iloc[index]["class_name"]

        if class_name not in class_found:
            sample = dataset[index]

            class_found[class_name] = {
                "shape": tuple(sample["input_values"].shape),
                "label": sample["label"].item()
            }

        if len(class_found) == len(CLASS_NAMES):
            break

    for class_name in CLASS_NAMES:
        if class_name in class_found:
            information = class_found[class_name]

            print(
                f"{class_name}: "
                f"shape={information['shape']}, "
                f"label={information['label']}"
            )
        else:
            print(f"{class_name}: NOT FOUND")

    print("\nClass counts:")

    counts = dataset.metadata["class_name"].value_counts()

    for class_name in CLASS_NAMES:
        print(
            f"{class_name}: "
            f"{counts.get(class_name, 0)}"
        )

    print("\nDataset sanity test completed.")