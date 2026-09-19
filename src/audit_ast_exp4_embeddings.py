from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from transformers import ASTForAudioClassification, AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
DATA_DIR = PROJECT_ROOT / "data/processed/unified_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ast_exp4/reports/ast_embedding_audit"

MODEL_CHECKPOINT = (
    PROJECT_ROOT /
    "outputs/ast_exp4/models/best_ast_exp4_model.pt"
)

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

TRAIN_CSV = DATA_DIR / "train_metadata.csv"
VAL_CSV = DATA_DIR / "validation_metadata.csv"
TEST_CSV = DATA_DIR / "test_metadata.csv"

TARGET_SR = 16000
TARGET_SECONDS = 4
TARGET_SAMPLES = TARGET_SR * TARGET_SECONDS

CLASS_NAMES = [
    "Asthma",
    "Bronchitis",
    "COPD",
    "Healthy",
    "Pneumonia"
]

DEVICE_NAME = "LittC2SE"
SOUND_CLASS = 0.0

PROBLEM_PATIENTS = {
    "ICBHI_134",
    "ICBHI_199"
}

CORRECT_PATIENTS = {
    "ICBHI_141"
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_metadata():
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    return train, val, test


def get_audio_path(row):
    path = Path(
        str(row["audio_path"])
    )

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_audio_id(audio_path):
    return Path(
        str(audio_path)
    ).stem


def get_device_from_audio_path(audio_path):
    name = Path(
        str(audio_path)
    ).stem

    if "_LittC2SE" in name:
        return "LittC2SE"

    if "_Litt3200" in name:
        return "Litt3200"

    if "_AKGC417L" in name:
        return "AKGC417L"

    if "_Meditron" in name:
        return "Meditron"

    return "UNKNOWN"


def load_audio(path):
    audio, sr = librosa.load(
        path,
        sr=TARGET_SR,
        mono=True
    )

    if len(audio) > TARGET_SAMPLES:
        start = (
            len(audio) - TARGET_SAMPLES
        ) // 2

        audio = audio[
            start:
            start + TARGET_SAMPLES
        ]

    elif len(audio) < TARGET_SAMPLES:
        audio = np.pad(
            audio,
            (
                0,
                TARGET_SAMPLES - len(audio)
            ),
            mode="constant"
        )

    return audio.astype(
        np.float32
    )


def filter_recordings(metadata):
    devices = metadata[
        "audio_path"
    ].apply(
        get_device_from_audio_path
    )

    mask = (
        (metadata["class_name"] == "COPD") &
        (
            metadata["sound_class"].astype(float)
            == SOUND_CLASS
        ) &
        (devices == DEVICE_NAME)
    )

    selected = metadata[
        mask
    ].copy()

    selected = selected.drop_duplicates(
        subset=["audio_path"]
    )

    return selected


def load_model():
    print("Loading AST feature extractor")

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    print("Loading Experiment 4 AST classifier")

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(
        MODEL_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    elif isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        state_dict = checkpoint

    print(
        f"Checkpoint parameters: "
        f"{len(state_dict)}"
    )

    model_state = model.state_dict()

    missing = [
        key
        for key in model_state
        if key not in state_dict
    ]

    unexpected = [
        key
        for key in state_dict
        if key not in model_state
    ]

    if missing:
        print()
        print("Missing checkpoint keys:")
        for key in missing[:20]:
            print(key)

    if unexpected:
        print()
        print("Unexpected checkpoint keys:")
        for key in unexpected[:20]:
            print(key)

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture does not match "
            "ASTForAudioClassification."
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.config.id2label = {
        index: name
        for index, name in enumerate(
            CLASS_NAMES
        )
    }

    model.config.label2id = {
        name: index
        for index, name in enumerate(
            CLASS_NAMES
        )
    }

    model.to(DEVICE)
    model.eval()

    print(
        f"Experiment 4 model loaded on {DEVICE}"
    )

    return (
        model,
        feature_extractor
    )


def process_recording(
    row,
    model,
    feature_extractor
):
    path = get_audio_path(row)

    audio = load_audio(
        path
    )

    processor_output = (
        feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        )
    )

    input_values = (
        processor_output[
            "input_values"
        ]
        .to(DEVICE)
    )

    with torch.no_grad():
        ast_outputs = (
            model.audio_spectrogram_transformer(
                input_values=input_values
            )
        )

        embedding = (
            ast_outputs.pooler_output
        )

        logits = model.classifier(
            embedding
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    embedding = (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
    )

    logits = (
        logits
        .squeeze(0)
        .cpu()
        .numpy()
    )

    probabilities = (
        probabilities
        .squeeze(0)
        .cpu()
        .numpy()
    )

    prediction_index = int(
        np.argmax(
            probabilities
        )
    )

    prediction = (
        CLASS_NAMES[
            prediction_index
        ]
    )

    true_class = str(
        row["class_name"]
    )

    audio_id = get_audio_id(
        row["audio_path"]
    )

    return {
        "split": row["split"],
        "patient_group": row[
            "patient_group"
        ],
        "audio_id": audio_id,
        "audio_path": row[
            "audio_path"
        ],
        "device": get_device_from_audio_path(
            row["audio_path"]
        ),
        "true_class": true_class,
        "prediction": prediction,
        "correct": (
            prediction == true_class
        ),
        "prob_asthma": float(
            probabilities[0]
        ),
        "prob_bronchitis": float(
            probabilities[1]
        ),
        "prob_copd": float(
            probabilities[2]
        ),
        "prob_healthy": float(
            probabilities[3]
        ),
        "prob_pneumonia": float(
            probabilities[4]
        ),
        "logit_asthma": float(
            logits[0]
        ),
        "logit_bronchitis": float(
            logits[1]
        ),
        "logit_copd": float(
            logits[2]
        ),
        "logit_healthy": float(
            logits[3]
        ),
        "logit_pneumonia": float(
            logits[4]
        ),
        "embedding": embedding
    }


def create_patient_embeddings(records):
    grouped = {}

    for record in records:
        patient = record[
            "patient_group"
        ]

        if patient not in grouped:
            grouped[patient] = []

        grouped[
            patient
        ].append(record)

    patient_records = []

    for patient, items in grouped.items():
        embeddings = np.vstack([
            item["embedding"]
            for item in items
        ])

        probabilities = np.vstack([
            np.array([
                item["prob_asthma"],
                item["prob_bronchitis"],
                item["prob_copd"],
                item["prob_healthy"],
                item["prob_pneumonia"]
            ])
            for item in items
        ])

        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        mean_probability = np.mean(
            probabilities,
            axis=0
        )

        prediction_index = int(
            np.argmax(
                mean_probability
            )
        )

        patient_records.append({
            "patient_group": patient,
            "split": items[0]["split"],
            "recordings": len(items),
            "mean_embedding": mean_embedding,
            "prob_asthma": float(
                mean_probability[0]
            ),
            "prob_bronchitis": float(
                mean_probability[1]
            ),
            "prob_copd": float(
                mean_probability[2]
            ),
            "prob_healthy": float(
                mean_probability[3]
            ),
            "prob_pneumonia": float(
                mean_probability[4]
            ),
            "mean_copd_pneumonia_margin": float(
                mean_probability[2]
                - mean_probability[4]
            ),
            "mean_prediction": CLASS_NAMES[
                prediction_index
            ]
        })

    return patient_records


def nearest_training_patients(
    patient_records
):
    train_patients = [
        p for p in patient_records
        if p["split"] == "train"
    ]

    target_patients = [
        p for p in patient_records
        if (
            p["patient_group"]
            in PROBLEM_PATIENTS
            or
            p["patient_group"]
            in CORRECT_PATIENTS
        )
    ]

    if not train_patients:
        return pd.DataFrame()

    train_embeddings = np.vstack([
        p["mean_embedding"]
        for p in train_patients
    ])

    rows = []

    for target in target_patients:
        similarities = cosine_similarity(
            target[
                "mean_embedding"
            ].reshape(1, -1),
            train_embeddings
        )[0]

        order = np.argsort(
            similarities
        )[::-1]

        for rank, index in enumerate(
            order[:10],
            start=1
        ):
            nearest = train_patients[
                index
            ]

            rows.append({
                "target_patient": target[
                    "patient_group"
                ],
                "rank": rank,
                "cosine_similarity": float(
                    similarities[index]
                ),
                "nearest_train_patient":
                    nearest[
                        "patient_group"
                    ],
                "nearest_train_recordings":
                    nearest[
                        "recordings"
                    ]
            })

    return pd.DataFrame(
        rows
    )


def compare_patient_embeddings(
    patient_records
):
    targets = [
        "ICBHI_141",
        "ICBHI_134",
        "ICBHI_199"
    ]

    selected = [
        p for p in patient_records
        if p["patient_group"]
        in targets
    ]

    rows = []

    for i in range(
        len(selected)
    ):
        for j in range(
            i + 1,
            len(selected)
        ):
            a = selected[i]
            b = selected[j]

            similarity = cosine_similarity(
                a[
                    "mean_embedding"
                ].reshape(1, -1),
                b[
                    "mean_embedding"
                ].reshape(1, -1)
            )[0, 0]

            rows.append({
                "patient_a": a[
                    "patient_group"
                ],
                "patient_b": b[
                    "patient_group"
                ],
                "cosine_similarity":
                    float(similarity)
            })

    return pd.DataFrame(
        rows
    )


def save_recording_results(
    records
):
    rows = []

    for record in records:
        row = {
            key: value
            for key, value in record.items()
            if key != "embedding"
        }

        rows.append(row)

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_DIR /
        "recording_level_predictions.csv",
        index=False
    )


def save_patient_results(
    patient_records
):
    rows = []

    for patient in patient_records:
        row = {
            key: value
            for key, value in patient.items()
            if key != "mean_embedding"
        }

        rows.append(row)

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_DIR /
        "patient_level_predictions.csv",
        index=False
    )


def save_embeddings(
    records
):
    rows = []

    for record in records:
        row = {
            "split": record["split"],
            "patient_group": record[
                "patient_group"
            ],
            "audio_id": record[
                "audio_id"
            ],
            "audio_path": record[
                "audio_path"
            ],
            "device": record[
                "device"
            ],
            "true_class": record[
                "true_class"
            ],
            "prediction": record[
                "prediction"
            ]
        }

        for i, value in enumerate(
            record["embedding"]
        ):
            row[
                f"embedding_{i + 1}"
            ] = float(value)

        rows.append(row)

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_DIR /
        "recording_embeddings.csv",
        index=False
    )


def print_patient_analysis(
    patient_records
):
    print()
    print(
        "PATIENT LEVEL MODEL OUTPUT"
    )

    for patient in patient_records:
        if (
            patient["patient_group"]
            not in (
                PROBLEM_PATIENTS
                | CORRECT_PATIENTS
            )
        ):
            continue

        print()
        print(
            patient["patient_group"]
        )
        print(
            f"Split: {patient['split']}"
        )
        print(
            f"Recordings: "
            f"{patient['recordings']}"
        )
        print(
            f"Asthma: "
            f"{patient['prob_asthma']:.4f}"
        )
        print(
            f"Bronchitis: "
            f"{patient['prob_bronchitis']:.4f}"
        )
        print(
            f"COPD: "
            f"{patient['prob_copd']:.4f}"
        )
        print(
            f"Healthy: "
            f"{patient['prob_healthy']:.4f}"
        )
        print(
            f"Pneumonia: "
            f"{patient['prob_pneumonia']:.4f}"
        )
        print(
            f"COPD minus Pneumonia: "
            f"{patient['mean_copd_pneumonia_margin']:.4f}"
        )
        print(
            f"Mean prediction: "
            f"{patient['mean_prediction']}"
        )


def print_recording_analysis(
    records
):
    print()
    print(
        "PROBLEM PATIENT RECORDING OUTPUT"
    )

    selected = [
        r for r in records
        if r["patient_group"]
        in (
            PROBLEM_PATIENTS
            | CORRECT_PATIENTS
        )
    ]

    for patient in [
        "ICBHI_141",
        "ICBHI_134",
        "ICBHI_199"
    ]:
        patient_records = [
            r for r in selected
            if r["patient_group"]
            == patient
        ]

        print()
        print(patient)

        for record in patient_records:
            print(
                f"{record['audio_id']} | "
                f"prediction="
                f"{record['prediction']} | "
                f"correct="
                f"{record['correct']} | "
                f"COPD="
                f"{record['prob_copd']:.4f} | "
                f"Pneumonia="
                f"{record['prob_pneumonia']:.4f}"
            )


def main():
    print(
        f"Using device: {DEVICE}"
    )

    train, val, test = (
        load_metadata()
    )

    selected_train = filter_recordings(
        train
    )

    selected_val = filter_recordings(
        val
    )

    selected_test = filter_recordings(
        test
    )

    print(
        f"Train recordings: "
        f"{len(selected_train)}"
    )

    print(
        f"Validation recordings: "
        f"{len(selected_val)}"
    )

    print(
        f"Test recordings: "
        f"{len(selected_test)}"
    )

    selected = pd.concat(
        [
            selected_train,
            selected_val,
            selected_test
        ],
        ignore_index=True
    )

    print(
        f"Total unique recordings: "
        f"{len(selected)}"
    )

    model, feature_extractor = (
        load_model()
    )

    records = []

    for _, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc="Running Experiment 4 model"
    ):
        try:
            result = process_recording(
                row,
                model,
                feature_extractor
            )

            records.append(
                result
            )

        except Exception as e:
            print()
            print(
                f"Failed: "
                f"{row['audio_path']}"
            )
            print(
                f"Error: {e}"
            )

    print(
        f"Successfully processed: "
        f"{len(records)}"
    )

    if not records:
        raise RuntimeError(
            "No recordings were successfully processed."
        )

    save_recording_results(
        records
    )

    save_embeddings(
        records
    )

    print(
        "Creating patient level embeddings"
    )

    patient_records = (
        create_patient_embeddings(
            records
        )
    )

    save_patient_results(
        patient_records
    )

    print_patient_analysis(
        patient_records
    )

    print_recording_analysis(
        records
    )

    print()
    print(
        "Calculating nearest training "
        "patients in learned embedding space"
    )

    nearest = (
        nearest_training_patients(
            patient_records
        )
    )

    nearest.to_csv(
        OUTPUT_DIR /
        "nearest_training_patients.csv",
        index=False
    )

    print()
    print(
        "NEAREST TRAINING PATIENTS"
    )

    if nearest.empty:
        print(
            "No training patient embeddings available."
        )
    else:
        for target in [
            "ICBHI_134",
            "ICBHI_199",
            "ICBHI_141"
        ]:
            subset = nearest[
                nearest[
                    "target_patient"
                ] == target
            ]

            print()
            print(target)

            for _, row in subset.head(5).iterrows():
                print(
                    f"{int(row['rank'])}. "
                    f"{row['nearest_train_patient']} "
                    f"similarity="
                    f"{row['cosine_similarity']:.4f}"
                )

    print()
    print(
        "PATIENT EMBEDDING SIMILARITY"
    )

    comparison = (
        compare_patient_embeddings(
            patient_records
        )
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        OUTPUT_DIR /
        "problem_patient_embedding_similarity.csv",
        index=False
    )

    print()
    print(
        "OUTPUT DIRECTORY"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()