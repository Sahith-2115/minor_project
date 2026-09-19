from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/home/sahith/projects/minor_project_408")
TRAIN_CSV = ROOT / "data/processed/unified_dataset/train_metadata.csv"
VAL_CSV = ROOT / "data/processed/unified_dataset/validation_metadata.csv"
TEST_CSV = ROOT / "data/processed/unified_dataset/test_metadata.csv"
CHECKPOINT = ROOT / "outputs/ast_exp4/models/best_ast_exp4_model.pt"
OUTPUT_DIR = ROOT / "outputs/ast_exp4/reports/ast_source_disease_audit"

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
AUDIO_SECONDS = 4
NUM_SAMPLES = SAMPLE_RATE * AUDIO_SECONDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["Asthma", "Bronchitis", "COPD", "Healthy", "Pneumonia"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

all_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

required_columns = [
    "source",
    "class_name",
    "patient_group",
    "audio_path",
    "split"
]

for column in required_columns:
    if column not in all_df.columns:
        raise RuntimeError(
            f"Required column '{column}' not found."
        )

def get_audio_id(audio_path):
    return Path(audio_path).stem

def load_audio(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio)),
            mode="constant"
        )
    elif len(audio) > NUM_SAMPLES:
        start = (len(audio) - NUM_SAMPLES) // 2
        audio = audio[
            start:start + NUM_SAMPLES
        ]

    return audio.astype(np.float32)

print("Loading AST feature extractor")

feature_extractor = ASTFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Loading Experiment 4 AST classifier")

model = ASTForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5,
    ignore_mismatched_sizes=True
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(DEVICE)
model.eval()

print(f"Checkpoint parameters: {len(state_dict)}")
print("Experiment 4 model loaded")

def process_recording(row):
    audio = load_audio(row["audio_path"])

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_values = inputs["input_values"].to(DEVICE)

    with torch.no_grad():
        outputs = model.audio_spectrogram_transformer(
            input_values=input_values
        )

        embedding = outputs.pooler_output

        logits = model.classifier(
            embedding
        )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

    embedding = embedding.squeeze(0).cpu().numpy()
    logits = logits.squeeze(0).cpu().numpy()
    probabilities = probabilities.squeeze(0).cpu().numpy()

    prediction = LABELS[
        int(np.argmax(probabilities))
    ]

    return {
        "audio_id": get_audio_id(
            row["audio_path"]
        ),
        "patient_group": row["patient_group"],
        "split": row["split"],
        "source": row["source"],
        "class_name": row["class_name"],
        "audio_path": row["audio_path"],
        "prediction": prediction,
        "embedding": embedding
    }

def extract_embeddings(df):
    records = []

    unique_df = df.drop_duplicates(
        subset=["audio_path"]
    ).copy()

    for _, row in tqdm(
        unique_df.iterrows(),
        total=len(unique_df),
        desc="Extracting embeddings"
    ):
        records.append(
            process_recording(row)
        )

    return records

print()
print("Preparing unique training recordings")

train_unique = train_df.drop_duplicates(
    subset=["audio_path"]
).copy()

print(
    f"Unique training recordings: "
    f"{len(train_unique)}"
)

print()
print("Extracting training embeddings")

train_records = extract_embeddings(
    train_unique
)

train_embedding_matrix = np.vstack(
    [
        record["embedding"]
        for record in train_records
    ]
)

train_record_df = pd.DataFrame(
    [
        {
            "audio_id": record["audio_id"],
            "patient_group": record["patient_group"],
            "split": record["split"],
            "source": record["source"],
            "class_name": record["class_name"],
            "audio_path": record["audio_path"],
            "prediction": record["prediction"]
        }
        for record in train_records
    ]
)

train_embedding_columns = [
    f"embedding_{i}"
    for i in range(
        train_embedding_matrix.shape[1]
    )
]

train_embedding_df = pd.DataFrame(
    train_embedding_matrix,
    columns=train_embedding_columns
)

train_full_df = pd.concat(
    [
        train_record_df.reset_index(drop=True),
        train_embedding_df.reset_index(drop=True)
    ],
    axis=1
)

train_full_df.to_csv(
    OUTPUT_DIR /
    "training_recording_embeddings.csv",
    index=False
)

print()
print("Creating training patient embeddings")

train_patient_df = (
    train_full_df
    .groupby(
        [
            "patient_group",
            "source",
            "class_name"
        ],
        as_index=False
    )[train_embedding_columns]
    .mean()
)

print(
    f"Training patient groups: "
    f"{len(train_patient_df)}"
)

train_patient_df.to_csv(
    OUTPUT_DIR /
    "training_patient_embeddings.csv",
    index=False
)

print()
print("Training patient distribution by source and class")

distribution = (
    train_patient_df
    .groupby(
        [
            "source",
            "class_name"
        ]
        )
    .size()
    .reset_index(
        name="patient_count"
    )
)

print(distribution.to_string(index=False))

distribution.to_csv(
    OUTPUT_DIR /
    "training_source_class_patient_counts.csv",
    index=False
)

print()
print("Extracting embeddings for target patients")

target_df = all_df[
    all_df["patient_group"].isin(
        [
            "ICBHI_141",
            "ICBHI_134",
            "ICBHI_199"
        ]
    )
].drop_duplicates(
    subset=["audio_path"]
).copy()

target_records = extract_embeddings(
    target_df
)

target_embedding_matrix = np.vstack(
    [
        record["embedding"]
        for record in target_records
    ]
)

target_record_df = pd.DataFrame(
    [
        {
            "audio_id": record["audio_id"],
            "patient_group": record["patient_group"],
            "split": record["split"],
            "source": record["source"],
            "class_name": record["class_name"],
            "audio_path": record["audio_path"],
            "prediction": record["prediction"]
        }
        for record in target_records
    ]
)

target_embedding_columns = [
    f"embedding_{i}"
    for i in range(
        target_embedding_matrix.shape[1]
    )
]

target_embedding_df = pd.DataFrame(
    target_embedding_matrix,
    columns=target_embedding_columns
)

target_full_df = pd.concat(
    [
        target_record_df.reset_index(drop=True),
        target_embedding_df.reset_index(drop=True)
    ],
    axis=1
)

target_full_df.to_csv(
    OUTPUT_DIR /
    "target_patient_recording_embeddings.csv",
    index=False
)

target_patient_df = (
    target_full_df
    .groupby(
        [
            "patient_group",
            "source",
            "class_name"
        ],
        as_index=False
    )[target_embedding_columns]
    .mean()
)

print()
print("Calculating source and disease centroids")

centroid_rows = []

for source in sorted(
    train_patient_df["source"].unique()
):

    for class_name in LABELS:

        subset = train_patient_df[
            (train_patient_df["source"] == source) &
            (train_patient_df["class_name"] == class_name)
        ]

        if len(subset) == 0:
            continue

        centroid = subset[
            train_embedding_columns
        ].mean().values

        centroid_rows.append(
            {
                "source": source,
                "class_name": class_name,
                "patient_count": len(subset),
                "centroid": centroid
            }
        )

centroid_matrix = np.vstack(
    [
        row["centroid"]
        for row in centroid_rows
    ]
)

centroid_meta = pd.DataFrame(
    [
        {
            "source": row["source"],
            "class_name": row["class_name"],
            "patient_count": row["patient_count"]
        }
        for row in centroid_rows
    ]
)

centroid_columns = [
    f"embedding_{i}"
    for i in range(
        centroid_matrix.shape[1]
    )
]

centroid_df = pd.concat(
    [
        centroid_meta,
        pd.DataFrame(
            centroid_matrix,
            columns=centroid_columns
        )
    ],
    axis=1
)

centroid_df.to_csv(
    OUTPUT_DIR /
    "source_disease_centroids.csv",
    index=False
)

print()
print("TARGET PATIENT DISTANCE TO SOURCE DISEASE CENTROIDS")

centroid_vectors = centroid_df[
    centroid_columns
].values

target_vectors = target_patient_df[
    target_embedding_columns
].values

target_centroid_similarity = cosine_similarity(
    target_vectors,
    centroid_vectors
)

distance_rows = []

for i, target_row in target_patient_df.iterrows():

    patient = target_row[
        "patient_group"
    ]

    print()
    print(patient)

    temp_rows = []

    for j, centroid_row in centroid_df.iterrows():

        similarity = target_centroid_similarity[
            i,
            j
        ]

        row = {
            "patient_group": patient,
            "target_source": target_row["source"],
            "target_class": target_row["class_name"],
            "reference_source": centroid_row["source"],
            "reference_class": centroid_row["class_name"],
            "patient_count": centroid_row["patient_count"],
            "cosine_similarity": float(similarity)
        }

        distance_rows.append(row)
        temp_rows.append(row)

    temp_df = pd.DataFrame(temp_rows)

    temp_df = temp_df.sort_values(
        "cosine_similarity",
        ascending=False
    )

    print(
        temp_df[
            [
                "reference_source",
                "reference_class",
                "cosine_similarity"
            ]
        ].head(10).to_string(
            index=False
        )
    )

distance_df = pd.DataFrame(
    distance_rows
)

distance_df.to_csv(
    OUTPUT_DIR /
    "target_patient_source_disease_centroid_similarity.csv",
    index=False
)

print()
print("TARGET PATIENT PAIRWISE SOURCE DISEASE ANALYSIS")

pair_rows = []

for _, target_row in target_patient_df.iterrows():

    patient = target_row[
        "patient_group"
    ]

    x = target_row[
        target_embedding_columns
    ].values.reshape(1, -1)

    print()
    print(patient)

    for source in sorted(
        train_patient_df["source"].unique()
    ):

        for class_name in LABELS:

            subset = train_patient_df[
                (train_patient_df["source"] == source) &
                (train_patient_df["class_name"] == class_name)
            ].copy()

            if len(subset) == 0:
                continue

            subset_vectors = subset[
                train_embedding_columns
            ].values

            similarities = cosine_similarity(
                x,
                subset_vectors
            )[0]

            nearest_index = np.argmax(
                similarities
            )

            nearest_similarity = (
                similarities[nearest_index]
            )

            nearest_patient = subset.iloc[
                nearest_index
            ]["patient_group"]

            mean_similarity = (
                similarities.mean()
            )

            median_similarity = (
                np.median(similarities)
            )

            top_k = min(
                5,
                len(similarities)
            )

            top5_mean = (
                np.sort(similarities)[
                    -top_k:
                ].mean()
            )

            pair_rows.append(
                {
                    "patient_group": patient,
                    "reference_source": source,
                    "reference_class": class_name,
                    "nearest_patient": nearest_patient,
                    "nearest_similarity": nearest_similarity,
                    "mean_similarity": mean_similarity,
                    "median_similarity": median_similarity,
                    "top5_mean_similarity": top5_mean
                }
            )

pairwise_df = pd.DataFrame(
    pair_rows
)

pairwise_df.to_csv(
    OUTPUT_DIR /
    "target_patient_source_disease_nearest_similarity.csv",
    index=False
)

print()
print("FOCUSED COMPARISON FOR PATIENT 134")

patient_134 = pairwise_df[
    pairwise_df["patient_group"] == "ICBHI_134"
].copy()

print(
    patient_134[
        [
            "reference_source",
            "reference_class",
            "nearest_patient",
            "nearest_similarity",
            "mean_similarity",
            "top5_mean_similarity"
        ]
    ]
    .sort_values(
        "nearest_similarity",
        ascending=False
    )
    .to_string(index=False)
)

print()
print("FOCUSED COMPARISON FOR PATIENT 141")

patient_141 = pairwise_df[
    pairwise_df["patient_group"] == "ICBHI_141"
].copy()

print(
    patient_141[
        [
            "reference_source",
            "reference_class",
            "nearest_patient",
            "nearest_similarity",
            "mean_similarity",
            "top5_mean_similarity"
        ]
    ]
    .sort_values(
        "nearest_similarity",
        ascending=False
    )
    .to_string(index=False)
)

print()
print("FOCUSED COMPARISON FOR PATIENT 199")

patient_199 = pairwise_df[
    pairwise_df["patient_group"] == "ICBHI_199"
].copy()

print(
    patient_199[
        [
            "reference_source",
            "reference_class",
            "nearest_patient",
            "nearest_similarity",
            "mean_similarity",
            "top5_mean_similarity"
        ]
    ]
    .sort_values(
        "nearest_similarity",
        ascending=False
    )
    .to_string(index=False)
)

print()
print("SOURCE SPECIFIC PNEUMONIA COMPARISON FOR PATIENT 134")

p134_pneumonia = pairwise_df[
    (pairwise_df["patient_group"] == "ICBHI_134") &
    (pairwise_df["reference_class"] == "Pneumonia")
].copy()

print(
    p134_pneumonia[
        [
            "reference_source",
            "nearest_patient",
            "nearest_similarity",
            "mean_similarity",
            "top5_mean_similarity"
        ]
    ]
    .sort_values(
        "nearest_similarity",
        ascending=False
    )
    .to_string(index=False)
)

print()
print("OUTPUT DIRECTORY")
print(OUTPUT_DIR)