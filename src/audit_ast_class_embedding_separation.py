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
OUTPUT_DIR = ROOT / "outputs/ast_exp4/reports/ast_class_embedding_audit"

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
AUDIO_SECONDS = 4
NUM_SAMPLES = SAMPLE_RATE * AUDIO_SECONDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["Asthma", "Bronchitis", "COPD", "Healthy", "Pneumonia"]
LABEL_TO_ID = {x: i for i, x in enumerate(LABELS)}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

all_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

print("Metadata columns:")
print(list(all_df.columns))

def get_device_from_path(audio_path):
    name = Path(audio_path).stem
    parts = name.split("_")
    if len(parts) >= 1 and parts[-1] in {
        "AKGC417L",
        "LittC2SE",
        "Litt3200",
        "Meditron"
    }:
        return parts[-1]
    return "UNKNOWN"

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
        audio = audio[start:start + NUM_SAMPLES]

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

required_columns = [
    "source",
    "class_name",
    "audio_path",
    "patient_group",
    "split"
]

for column in required_columns:
    if column not in all_df.columns:
        raise RuntimeError(
            f"Expected column '{column}' was not found in metadata."
        )

target_df = all_df[
    (all_df["source"] == "ICBHI") &
    (all_df["class_name"] == "COPD") &
    (all_df["audio_path"].str.contains("LittC2SE", na=False)) &
    (all_df["sound_class"] == 0)
].copy()

target_df["audio_id"] = target_df["audio_path"].apply(
    get_audio_id
)

target_df = target_df.drop_duplicates(
    subset=["audio_path"]
).copy()

problem_target = target_df[
    target_df["patient_group"].isin(
        ["ICBHI_134", "ICBHI_199", "ICBHI_141"]
    )
].copy()

print(
    f"Selected LittC2SE COPD class 0 recordings: "
    f"{len(target_df)}"
)

print(
    f"Problem/control recordings selected: "
    f"{len(problem_target)}"
)

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

    prediction_id = int(
        np.argmax(probabilities)
    )

    result = {
        "audio_id": row["audio_id"],
        "patient_group": row["patient_group"],
        "split": row["split"],
        "class_name": row["class_name"],
        "source": row["source"],
        "audio_path": row["audio_path"],
        "device": get_device_from_path(
            row["audio_path"]
        ),
        "prediction": LABELS[prediction_id]
    }

    for i, label in enumerate(LABELS):
        result[f"logit_{label}"] = float(
            logits[i]
        )
        result[f"prob_{label}"] = float(
            probabilities[i]
        )

    result["copd_pneumonia_logit_margin"] = float(
        logits[LABEL_TO_ID["COPD"]] -
        logits[LABEL_TO_ID["Pneumonia"]]
    )

    result["copd_pneumonia_probability_margin"] = float(
        probabilities[LABEL_TO_ID["COPD"]] -
        probabilities[LABEL_TO_ID["Pneumonia"]]
    )

    return result, embedding

def extract_embeddings(df, description):
    records = []
    embeddings = []

    df = df.drop_duplicates(
        subset=["audio_path"]
    ).copy()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=description
    ):
        result, embedding = process_recording(row)
        records.append(result)
        embeddings.append(embedding)

    if len(embeddings) == 0:
        return pd.DataFrame(), np.empty(
            (0, 768),
            dtype=np.float32
        )

    return (
        pd.DataFrame(records),
        np.vstack(embeddings)
    )

print()
print("Extracting embeddings for problem/control recordings")

problem_predictions, problem_embeddings = extract_embeddings(
    problem_target,
    "Problem/control embeddings"
)

problem_embedding_columns = [
    f"embedding_{i}"
    for i in range(
        problem_embeddings.shape[1]
    )
]

problem_embedding_df = pd.DataFrame(
    problem_embeddings,
    columns=problem_embedding_columns
)

problem_full_df = pd.concat(
    [
        problem_predictions.reset_index(drop=True),
        problem_embedding_df.reset_index(drop=True)
    ],
    axis=1
)

problem_full_df.to_csv(
    OUTPUT_DIR /
    "problem_control_recordings.csv",
    index=False
)

print()
print("Preparing complete training patient embedding database")

train_all = train_df.copy()

train_all["audio_id"] = train_all[
    "audio_path"
].apply(get_audio_id)

train_all = train_all.drop_duplicates(
    subset=["audio_path"]
).copy()

print(
    f"Training metadata recordings: "
    f"{len(train_all)}"
)

print(
    "Extracting embeddings for all unique training recordings"
)

train_predictions, train_embeddings = extract_embeddings(
    train_all,
    "Training embeddings"
)

train_embedding_columns = [
    f"embedding_{i}"
    for i in range(
        train_embeddings.shape[1]
    )
]

train_embedding_df = pd.DataFrame(
    train_embeddings,
    columns=train_embedding_columns
)

train_full_df = pd.concat(
    [
        train_predictions.reset_index(drop=True),
        train_embedding_df.reset_index(drop=True)
    ],
    axis=1
)

train_full_df.to_csv(
    OUTPUT_DIR /
    "all_training_recording_embeddings.csv",
    index=False
)

print()
print("Creating patient level training embeddings")

train_patient_embeddings = (
    train_full_df
    .groupby(
        [
            "patient_group",
            "class_name"
        ],
        as_index=False
    )[train_embedding_columns]
    .mean()
)

print(
    f"Training patient embeddings: "
    f"{len(train_patient_embeddings)}"
)

print()
print("Training patient class distribution")

print(
    train_patient_embeddings[
        "class_name"
    ].value_counts()
)

problem_patient_embeddings = (
    problem_full_df
    .groupby(
        [
            "patient_group",
            "class_name"
        ],
        as_index=False
    )[problem_embedding_columns]
    .mean()
)

print()
print("CLASS SPECIFIC EMBEDDING ANALYSIS")

class_results = []
similarity_rows = []

X_train = train_patient_embeddings[
    train_embedding_columns
].values

for _, problem_row in problem_patient_embeddings.iterrows():

    patient = problem_row[
        "patient_group"
    ]

    x = problem_row[
        problem_embedding_columns
    ].values.reshape(1, -1)

    train_sims = cosine_similarity(
        x,
        X_train
    )[0]

    temp = train_patient_embeddings.copy()
    temp["similarity"] = train_sims
    temp["query_patient"] = patient

    similarity_rows.append(
        temp[
            [
                "query_patient",
                "patient_group",
                "class_name",
                "similarity"
            ]
        ].copy()
    )

    print()
    print(patient)

    for class_name in [
        "COPD",
        "Pneumonia"
    ]:

        class_patients = temp[
            temp["class_name"] == class_name
        ].copy()

        if len(class_patients) == 0:
            print(
                f"{class_name}: no training patients found"
            )
            continue

        values = class_patients[
            "similarity"
        ].values

        nearest_index = class_patients[
            "similarity"
        ].idxmax()

        nearest_patient = class_patients.loc[
            nearest_index,
            "patient_group"
        ]

        nearest_similarity = (
            class_patients[
                "similarity"
            ].max()
        )

        mean_similarity = values.mean()
        median_similarity = np.median(values)

        top_k = min(
            5,
            len(values)
        )

        top_k_mean = (
            class_patients
            .nlargest(
                top_k,
                "similarity"
            )["similarity"]
            .mean()
        )

        print(
            f"{class_name}: "
            f"nearest={nearest_patient} "
            f"similarity={nearest_similarity:.4f}, "
            f"mean={mean_similarity:.4f}, "
            f"median={median_similarity:.4f}, "
            f"top{top_k}_mean={top_k_mean:.4f}"
        )

        class_results.append(
            {
                "problem_patient": patient,
                "comparison_class": class_name,
                "nearest_patient": nearest_patient,
                "nearest_similarity": nearest_similarity,
                "mean_similarity": mean_similarity,
                "median_similarity": median_similarity,
                "top_k_mean_similarity": top_k_mean
            }
        )

similarity_df = pd.concat(
    similarity_rows,
    ignore_index=True
)

similarity_df = similarity_df.sort_values(
    [
        "query_patient",
        "similarity"
    ],
    ascending=[
        True,
        False
    ]
)

similarity_df.to_csv(
    OUTPUT_DIR /
    "problem_patient_training_similarity.csv",
    index=False
)

class_similarity_df = pd.DataFrame(
    class_results
)

class_similarity_df.to_csv(
    OUTPUT_DIR /
    "problem_patient_class_similarity.csv",
    index=False
)

print()
print("CLASS SIMILARITY DIFFERENCE")

for patient in [
    "ICBHI_141",
    "ICBHI_134",
    "ICBHI_199"
]:

    temp = class_similarity_df[
        class_similarity_df[
            "problem_patient"
        ] == patient
    ]

    copd_rows = temp[
        temp["comparison_class"] == "COPD"
    ]

    pneumonia_rows = temp[
        temp["comparison_class"] == "Pneumonia"
    ]

    if len(copd_rows) == 0 or len(pneumonia_rows) == 0:
        continue

    copd = copd_rows.iloc[0]
    pneumonia = pneumonia_rows.iloc[0]

    print()
    print(patient)

    print(
        "Nearest similarity difference "
        "COPD minus Pneumonia: "
        f"{copd['nearest_similarity'] - pneumonia['nearest_similarity']:.4f}"
    )

    print(
        "Mean similarity difference "
        "COPD minus Pneumonia: "
        f"{copd['mean_similarity'] - pneumonia['mean_similarity']:.4f}"
    )

    print(
        "Top 5 similarity difference "
        "COPD minus Pneumonia: "
        f"{copd['top_k_mean_similarity'] - pneumonia['top_k_mean_similarity']:.4f}"
    )

print()
print("MODEL LOGIT ANALYSIS")

for patient in [
    "ICBHI_141",
    "ICBHI_134",
    "ICBHI_199"
]:

    temp = problem_full_df[
        problem_full_df[
            "patient_group"
        ] == patient
    ]

    if len(temp) == 0:
        continue

    mean_logits = temp[
        [
            "logit_COPD",
            "logit_Pneumonia"
        ]
    ].mean()

    mean_probs = temp[
        [
            "prob_COPD",
            "prob_Pneumonia"
        ]
    ].mean()

    margin = (
        mean_logits["logit_COPD"] -
        mean_logits["logit_Pneumonia"]
    )

    print()
    print(patient)

    print(
        f"Mean COPD logit: "
        f"{mean_logits['logit_COPD']:.4f}"
    )

    print(
        f"Mean Pneumonia logit: "
        f"{mean_logits['logit_Pneumonia']:.4f}"
    )

    print(
        "COPD minus Pneumonia logit margin: "
        f"{margin:.4f}"
    )

    print(
        f"Mean COPD probability: "
        f"{mean_probs['prob_COPD']:.4f}"
    )

    print(
        f"Mean Pneumonia probability: "
        f"{mean_probs['prob_Pneumonia']:.4f}"
    )

print()
print("TOP TRAINING PATIENTS BY CLASS")

for patient in [
    "ICBHI_141",
    "ICBHI_134",
    "ICBHI_199"
]:

    print()
    print(patient)

    temp = similarity_df[
        similarity_df[
            "query_patient"
        ] == patient
    ].copy()

    for class_name in [
        "COPD",
        "Pneumonia"
    ]:

        print()
        print(
            f"Nearest training {class_name} patients:"
        )

        class_temp = temp[
            temp["class_name"] == class_name
        ].head(5)

        for rank, (_, row) in enumerate(
            class_temp.iterrows(),
            start=1
        ):
            print(
                f"{rank}. "
                f"{row['patient_group']} "
                f"similarity={row['similarity']:.4f}"
            )

print()
print("OUTPUT DIRECTORY")
print(OUTPUT_DIR)