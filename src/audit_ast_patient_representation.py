from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
DATA_DIR = PROJECT_ROOT / "data/processed/unified_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ast_exp4/reports/ast_patient_audit"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

TRAIN_CSV = DATA_DIR / "train_metadata.csv"
VAL_CSV = DATA_DIR / "validation_metadata.csv"
TEST_CSV = DATA_DIR / "test_metadata.csv"

TARGET_SR = 16000
TARGET_SECONDS = 4
TARGET_SAMPLES = TARGET_SR * TARGET_SECONDS

CLASS_NAME = "COPD"
SOUND_CLASS = 0.0
DEVICE_NAME = "LittC2SE"

PROBLEM_PATIENTS = {
    "ICBHI_134",
    "ICBHI_199"
}

CORRECT_PATIENTS = {
    "ICBHI_141"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata():
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    return train, val, test


def get_audio_path(row):
    path = Path(str(row["audio_path"]))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_filename(row):
    return Path(str(row["audio_path"])).name


def get_device_from_audio_path(audio_path):
    filename = Path(str(audio_path)).name

    device_names = [
        "AKGC417L",
        "LittC2SE",
        "Litt3200",
        "Meditron"
    ]

    for device in device_names:
        if device in filename:
            return device

    return "Unknown"


def load_audio(path):
    audio, sr = librosa.load(
        path,
        sr=TARGET_SR,
        mono=True
    )

    if len(audio) > TARGET_SAMPLES:
        start = (len(audio) - TARGET_SAMPLES) // 2
        audio = audio[start:start + TARGET_SAMPLES]

    elif len(audio) < TARGET_SAMPLES:
        audio = np.pad(
            audio,
            (0, TARGET_SAMPLES - len(audio)),
            mode="constant"
        )

    return audio.astype(np.float32)


def extract_ast_input(audio, feature_extractor):
    result = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="np"
    )

    return result["input_values"][0].astype(np.float32)


def extract_features(ast_input):
    low32 = ast_input[:, :32]
    low64 = ast_input[:, :64]
    high64 = ast_input[:, 64:]
    high32 = ast_input[:, 96:]

    low32_mean = float(np.mean(low32))
    low64_mean = float(np.mean(low64))
    high64_mean = float(np.mean(high64))
    high32_mean = float(np.mean(high32))

    return {
        "ast_mean": float(np.mean(ast_input)),
        "ast_std": float(np.std(ast_input)),
        "ast_min": float(np.min(ast_input)),
        "ast_max": float(np.max(ast_input)),
        "low32_mean": low32_mean,
        "low64_mean": low64_mean,
        "high64_mean": high64_mean,
        "high32_mean": high32_mean,
        "low_high_difference": low64_mean - high64_mean
    }


def filter_recordings(metadata):
    metadata = metadata.copy()

    metadata["derived_device"] = metadata[
        "audio_path"
    ].apply(get_device_from_audio_path)

    mask = (
        (metadata["class_name"] == CLASS_NAME) &
        (metadata["sound_class"].astype(float) == SOUND_CLASS) &
        (metadata["derived_device"] == DEVICE_NAME)
    )

    selected = metadata[mask].copy()

    selected = selected.drop_duplicates(
        subset=["audio_path"]
    )

    return selected


def collect_recordings(metadata, feature_extractor):
    records = []

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Extracting recording representations"
    ):
        try:
            path = get_audio_path(row)

            if not path.exists():
                print(f"Missing audio: {path}")
                continue

            audio = load_audio(path)

            ast_input = extract_ast_input(
                audio,
                feature_extractor
            )

            vector = ast_input.reshape(-1)

            features = extract_features(ast_input)

            record = {
                "split": row["split"],
                "patient_group": row["patient_group"],
                "class_name": row["class_name"],
                "sound_class": row["sound_class"],
                "audio_path": row["audio_path"],
                "audio_filename": get_filename(row),
                "device": get_device_from_audio_path(
                    row["audio_path"]
                ),
                "ast_vector": vector
            }

            record.update(features)

            records.append(record)

        except Exception as e:
            print(f"Failed: {row['audio_path']}")
            print(f"Error: {e}")

    return records


def create_patient_centroids(records):
    patient_rows = []

    grouped = {}

    for record in records:
        patient = record["patient_group"]

        if patient not in grouped:
            grouped[patient] = []

        grouped[patient].append(record)

    for patient, patient_records in grouped.items():
        vectors = np.vstack([
            r["ast_vector"]
            for r in patient_records
        ])

        centroid = np.mean(
            vectors,
            axis=0
        )

        patient_rows.append({
            "patient_group": patient,
            "split": patient_records[0]["split"],
            "recordings": len(patient_records),
            "ast_mean": float(np.mean(centroid)),
            "ast_std": float(np.std(centroid)),
            "ast_vector": centroid
        })

    return patient_rows


def calculate_patient_similarity(patient_records):
    train_patients = [
        p for p in patient_records
        if p["split"] == "train"
    ]

    target_patients = [
        p for p in patient_records
        if (
            p["patient_group"] in PROBLEM_PATIENTS
            or p["patient_group"] in CORRECT_PATIENTS
        )
    ]

    if not train_patients:
        return pd.DataFrame()

    train_vectors = np.vstack([
        p["ast_vector"]
        for p in train_patients
    ])

    rows = []

    for target in target_patients:
        similarities = cosine_similarity(
            target["ast_vector"].reshape(1, -1),
            train_vectors
        )[0]

        order = np.argsort(similarities)[::-1]

        for rank, index in enumerate(
            order[:10],
            start=1
        ):
            nearest = train_patients[index]

            rows.append({
                "target_patient": target["patient_group"],
                "target_split": target["split"],
                "rank": rank,
                "cosine_similarity": float(
                    similarities[index]
                ),
                "nearest_train_patient": nearest["patient_group"],
                "nearest_train_recordings": nearest["recordings"]
            })

    return pd.DataFrame(rows)


def calculate_group_centroid(records):
    vectors = np.vstack([
        r["ast_vector"]
        for r in records
    ])

    return np.mean(
        vectors,
        axis=0
    )


def calculate_group_similarity(records):
    groups = {
        "training_recordings": [
            r for r in records
            if r["split"] == "train"
        ],
        "patient_141": [
            r for r in records
            if r["patient_group"] == "ICBHI_141"
        ],
        "patient_134": [
            r for r in records
            if r["patient_group"] == "ICBHI_134"
        ],
        "patient_199": [
            r for r in records
            if r["patient_group"] == "ICBHI_199"
        ]
    }

    centroids = {}

    for name, group in groups.items():
        if group:
            centroids[name] = calculate_group_centroid(group)

    rows = []

    target_names = [
        "patient_141",
        "patient_134",
        "patient_199"
    ]

    for target_name in target_names:
        if target_name not in centroids:
            continue

        target = centroids[target_name]

        for reference_name, reference in centroids.items():
            if reference_name == target_name:
                continue

            similarity = cosine_similarity(
                target.reshape(1, -1),
                reference.reshape(1, -1)
            )[0, 0]

            rows.append({
                "target": target_name,
                "reference": reference_name,
                "cosine_similarity": float(similarity)
            })

    return pd.DataFrame(rows)


def create_pca(records):
    vectors = np.vstack([
        r["ast_vector"]
        for r in records
    ])

    scaler = StandardScaler()

    scaled = scaler.fit_transform(vectors)

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(scaled)

    rows = []

    for i, record in enumerate(records):
        rows.append({
            "split": record["split"],
            "patient_group": record["patient_group"],
            "audio_filename": record["audio_filename"],
            "audio_path": record["audio_path"],
            "pc1": float(components[i, 0]),
            "pc2": float(components[i, 1])
        })

    explained = pd.DataFrame({
        "component": [
            "PC1",
            "PC2"
        ],
        "explained_variance_ratio": pca.explained_variance_ratio_
    })

    return pd.DataFrame(rows), explained


def create_patient_pca(patient_records):
    vectors = np.vstack([
        p["ast_vector"]
        for p in patient_records
    ])

    scaler = StandardScaler()

    scaled = scaler.fit_transform(vectors)

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(scaled)

    rows = []

    for i, patient in enumerate(patient_records):
        rows.append({
            "patient_group": patient["patient_group"],
            "split": patient["split"],
            "recordings": patient["recordings"],
            "pc1": float(components[i, 0]),
            "pc2": float(components[i, 1])
        })

    explained = pd.DataFrame({
        "component": [
            "PC1",
            "PC2"
        ],
        "explained_variance_ratio": pca.explained_variance_ratio_
    })

    return pd.DataFrame(rows), explained


def main():
    print("Loading metadata")

    train, val, test = load_metadata()

    selected_train = filter_recordings(train)
    selected_val = filter_recordings(val)
    selected_test = filter_recordings(test)

    print(f"Train recordings: {len(selected_train)}")
    print(f"Validation recordings: {len(selected_val)}")
    print(f"Test recordings: {len(selected_test)}")

    selected = pd.concat(
        [
            selected_train,
            selected_val,
            selected_test
        ],
        ignore_index=True
    )

    print(f"Total unique recordings: {len(selected)}")

    print("Loading AST feature extractor")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    records = collect_recordings(
        selected,
        feature_extractor
    )

    print(f"Successfully processed: {len(records)}")

    recording_rows = []

    for record in records:
        row = {
            key: value
            for key, value in record.items()
            if key != "ast_vector"
        }

        recording_rows.append(row)

    pd.DataFrame(recording_rows).to_csv(
        OUTPUT_DIR / "recording_level_ast_features.csv",
        index=False
    )

    print("Creating patient centroids")

    patient_records = create_patient_centroids(
        records
    )

    patient_rows = []

    for patient in patient_records:
        row = {
            key: value
            for key, value in patient.items()
            if key != "ast_vector"
        }

        patient_rows.append(row)

    pd.DataFrame(patient_rows).to_csv(
        OUTPUT_DIR / "patient_level_ast_features.csv",
        index=False
    )

    print("Calculating nearest training patients")

    similarity_df = calculate_patient_similarity(
        patient_records
    )

    similarity_df.to_csv(
        OUTPUT_DIR / "nearest_training_patients.csv",
        index=False
    )

    print("Calculating group similarities")

    group_similarity = calculate_group_similarity(
        records
    )

    group_similarity.to_csv(
        OUTPUT_DIR / "group_similarity.csv",
        index=False
    )

    print("Creating recording level PCA")

    recording_pca, recording_explained = create_pca(
        records
    )

    recording_pca.to_csv(
        OUTPUT_DIR / "recording_level_pca.csv",
        index=False
    )

    recording_explained.to_csv(
        OUTPUT_DIR / "recording_pca_explained_variance.csv",
        index=False
    )

    print("Creating patient level PCA")

    patient_pca, patient_explained = create_patient_pca(
        patient_records
    )

    patient_pca.to_csv(
        OUTPUT_DIR / "patient_level_pca.csv",
        index=False
    )

    patient_explained.to_csv(
        OUTPUT_DIR / "patient_pca_explained_variance.csv",
        index=False
    )

    print()
    print("PATIENT LEVEL SUMMARY")

    for patient in patient_records:
        if (
            patient["patient_group"] in PROBLEM_PATIENTS
            or patient["patient_group"] in CORRECT_PATIENTS
        ):
            print(
                f"{patient['patient_group']}: "
                f"split={patient['split']}, "
                f"recordings={patient['recordings']}, "
                f"mean={patient['ast_mean']:.4f}, "
                f"std={patient['ast_std']:.4f}"
            )

    print()
    print("NEAREST TRAINING PATIENTS")

    for target in [
        "ICBHI_134",
        "ICBHI_199",
        "ICBHI_141"
    ]:
        subset = similarity_df[
            similarity_df["target_patient"] == target
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
    print("GROUP LEVEL SIMILARITY")

    if not group_similarity.empty:
        print(
            group_similarity.to_string(
                index=False
            )
        )

    print()
    print("PCA EXPLAINED VARIANCE")

    print(
        recording_explained.to_string(
            index=False
        )
    )

    print()
    print("OUTPUT DIRECTORY")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()