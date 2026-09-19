from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path("/home/sahith/projects/minor_project_408")
DATA_DIR = PROJECT_ROOT / "data/processed/unified_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ast_exp4/reports/ast_input_audit"

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

def get_audio_id(audio_path):
    return Path(str(audio_path)).stem

def extract_device(audio_path):
    name = Path(str(audio_path)).stem
    parts = name.split("_")

    if len(parts) >= 5:
        return parts[4]

    return "UNKNOWN"

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
    low = ast_input[:, :32]
    low_mid = ast_input[:, 32:64]
    high_mid = ast_input[:, 64:96]
    high = ast_input[:, 96:]

    low_mean = float(np.mean(low))
    low_mid_mean = float(np.mean(low_mid))
    high_mid_mean = float(np.mean(high_mid))
    high_mean = float(np.mean(high))

    low_half = float(np.mean(ast_input[:, :64]))
    high_half = float(np.mean(ast_input[:, 64:]))

    return {
        "ast_mean": float(np.mean(ast_input)),
        "ast_std": float(np.std(ast_input)),
        "ast_min": float(np.min(ast_input)),
        "ast_max": float(np.max(ast_input)),
        "low_32_mean": low_mean,
        "low_mid_32_mean": low_mid_mean,
        "high_mid_32_mean": high_mid_mean,
        "high_32_mean": high_mean,
        "low_high_difference": low_mean - high_mean,
        "low_half_mean": low_half,
        "high_half_mean": high_half,
        "low_half_high_half_difference": low_half - high_half
    }

def make_frequency_profile(ast_input):
    return np.mean(ast_input, axis=0)

def build_record(row, ast_input):
    features = extract_features(ast_input)

    record = {
        "split": row["split"],
        "patient_group": row["patient_group"],
        "class_name": row["class_name"],
        "sound_class": row["sound_class"],
        "audio_path": row["audio_path"],
        "audio_id": get_audio_id(row["audio_path"]),
        "device": extract_device(row["audio_path"])
    }

    record.update(features)

    return record

def collect_records(metadata, feature_extractor):
    rows = []

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Extracting AST representations"
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

            record = build_record(
                row,
                ast_input
            )

            record["_ast_vector"] = ast_input.reshape(-1)
            record["_mel_profile"] = make_frequency_profile(
                ast_input
            )

            rows.append(record)

        except Exception as e:
            print(f"Failed: {row['audio_path']}")
            print(f"Error: {e}")

    return rows

def filter_littc2se_copd(metadata):
    devices = metadata["audio_path"].apply(
        extract_device
    )

    sound_classes = pd.to_numeric(
        metadata["sound_class"],
        errors="coerce"
    )

    mask = (
        (metadata["class_name"] == CLASS_NAME) &
        (sound_classes == SOUND_CLASS) &
        (devices == DEVICE_NAME)
    )

    return metadata[mask].copy()

def save_frequency_profiles(records):
    profile_rows = []

    for record in records:
        profile = record["_mel_profile"]

        row = {
            "split": record["split"],
            "patient_group": record["patient_group"],
            "audio_id": record["audio_id"],
            "audio_path": record["audio_path"],
            "device": record["device"]
        }

        for i, value in enumerate(profile):
            row[f"mel_bin_{i + 1}"] = float(value)

        profile_rows.append(row)

    output = pd.DataFrame(profile_rows)

    output.to_csv(
        OUTPUT_DIR / "ast_mel_frequency_profiles.csv",
        index=False
    )

def calculate_group_profile(records):
    vectors = np.vstack([
        r["_ast_vector"]
        for r in records
    ])

    profiles = np.vstack([
        r["_mel_profile"]
        for r in records
    ])

    return {
        "vector_mean": np.mean(vectors, axis=0),
        "profile_mean": np.mean(profiles, axis=0)
    }

def add_similarity_to_training(records):
    train_records = [
        r for r in records
        if r["split"] == "train"
    ]

    problem_records = [
        r for r in records
        if r["patient_group"] in {
            "ICBHI_134",
            "ICBHI_199"
        }
    ]

    if not train_records or not problem_records:
        return pd.DataFrame()

    train_vectors = np.vstack([
        r["_ast_vector"]
        for r in train_records
    ])

    problem_vectors = np.vstack([
        r["_ast_vector"]
        for r in problem_records
    ])

    similarities = cosine_similarity(
        problem_vectors,
        train_vectors
    )

    rows = []

    for i, problem in enumerate(problem_records):
        order = np.argsort(
            similarities[i]
        )[::-1]

        for rank, idx in enumerate(
            order[:10],
            start=1
        ):
            train_record = train_records[idx]

            rows.append({
                "problem_patient": problem["patient_group"],
                "problem_audio_id": problem["audio_id"],
                "problem_audio_path": problem["audio_path"],
                "rank": rank,
                "cosine_similarity": float(
                    similarities[i, idx]
                ),
                "training_patient": train_record["patient_group"],
                "training_audio_id": train_record["audio_id"],
                "training_audio_path": train_record["audio_path"]
            })

    return pd.DataFrame(rows)

def print_group_summary(name, records):
    if not records:
        print(f"{name}: 0 recordings")
        return

    ast_mean = np.mean([
        r["ast_mean"]
        for r in records
    ])

    ast_std = np.mean([
        r["ast_std"]
        for r in records
    ])

    low = np.mean([
        r["low_half_mean"]
        for r in records
    ])

    high = np.mean([
        r["high_half_mean"]
        for r in records
    ])

    print(
        f"{name}: "
        f"n={len(records)}, "
        f"mean={ast_mean:.4f}, "
        f"std={ast_std:.4f}, "
        f"low64={low:.4f}, "
        f"high64={high:.4f}, "
        f"low_high={low - high:.4f}"
    )

def main():
    print("Loading metadata")

    train, val, test = load_metadata()

    selected_train = filter_littc2se_copd(
        train
    )

    selected_val = filter_littc2se_copd(
        val
    )

    selected_test = filter_littc2se_copd(
        test
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
        f"Selected LittC2SE COPD class 0 recordings: "
        f"{len(selected)}"
    )

    print(
        f"Train: {len(selected_train)}"
    )

    print(
        f"Validation: {len(selected_val)}"
    )

    print(
        f"Test: {len(selected_test)}"
    )

    print("Loading AST feature extractor")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    records = collect_records(
        selected,
        feature_extractor
    )

    print(
        f"Successfully processed: {len(records)}"
    )

    if not records:
        raise RuntimeError(
            "No recordings were processed"
        )

    simple_records = []

    for record in records:
        clean = {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }

        simple_records.append(clean)

    pd.DataFrame(
        simple_records
    ).to_csv(
        OUTPUT_DIR /
        "ast_input_recording_features.csv",
        index=False
    )

    save_frequency_profiles(records)

    training_records = [
        r for r in records
        if r["split"] == "train"
    ]

    patient_134 = [
        r for r in records
        if r["patient_group"] == "ICBHI_134"
    ]

    patient_199 = [
        r for r in records
        if r["patient_group"] == "ICBHI_199"
    ]

    patient_141 = [
        r for r in records
        if r["patient_group"] == "ICBHI_141"
    ]

    print()
    print("AST INPUT GROUP SUMMARY")

    print_group_summary(
        "Training LittC2SE COPD class 0",
        training_records
    )

    print_group_summary(
        "Validation patient 141",
        patient_141
    )

    print_group_summary(
        "Test patient 134",
        patient_134
    )

    print_group_summary(
        "Validation patient 199",
        patient_199
    )

    similarity_df = add_similarity_to_training(
        records
    )

    if not similarity_df.empty:
        similarity_df.to_csv(
            OUTPUT_DIR /
            "problem_patient_nearest_training_recordings.csv",
            index=False
        )

    groups = {
        "training": training_records,
        "patient_141": patient_141,
        "patient_134": patient_134,
        "patient_199": patient_199
    }

    group_rows = []

    for name, group in groups.items():
        if not group:
            continue

        profile = calculate_group_profile(
            group
        )

        group_rows.append({
            "group": name,
            "recordings": len(group),
            "mean_ast_value": float(
                np.mean(profile["vector_mean"])
            ),
            "std_ast_value": float(
                np.std(profile["vector_mean"])
            ),
            "low_32_mean": float(
                np.mean(
                    profile["profile_mean"][:32]
                )
            ),
            "low_64_mean": float(
                np.mean(
                    profile["profile_mean"][:64]
                )
            ),
            "high_64_mean": float(
                np.mean(
                    profile["profile_mean"][64:]
                )
            ),
            "high_32_mean": float(
                np.mean(
                    profile["profile_mean"][96:]
                )
            ),
            "low_high_difference": float(
                np.mean(
                    profile["profile_mean"][:64]
                ) -
                np.mean(
                    profile["profile_mean"][64:]
                )
            )
        })

    pd.DataFrame(
        group_rows
    ).to_csv(
        OUTPUT_DIR /
        "ast_input_group_comparison.csv",
        index=False
    )

    print()
    print("OUTPUT FILES")

    print(
        OUTPUT_DIR /
        "ast_input_recording_features.csv"
    )

    print(
        OUTPUT_DIR /
        "ast_mel_frequency_profiles.csv"
    )

    print(
        OUTPUT_DIR /
        "ast_input_group_comparison.csv"
    )

    print(
        OUTPUT_DIR /
        "problem_patient_nearest_training_recordings.csv"
    )

    if not similarity_df.empty:
        print()
        print("NEAREST TRAINING RECORDINGS")

        for patient in [
            "ICBHI_134",
            "ICBHI_199"
        ]:
            subset = similarity_df[
                similarity_df["problem_patient"] == patient
            ]

            print()
            print(patient)

            for _, row in subset.head(5).iterrows():
                print(
                    f"{row['rank']}. "
                    f"{row['training_patient']} "
                    f"{row['training_audio_id']} "
                    f"similarity="
                    f"{row['cosine_similarity']:.4f}"
                )

if __name__ == "__main__":
    main()