import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from sklearn.model_selection import train_test_split

from data_loader import load_icbhi_cycles
from preprocessing import preprocess_audio_from_signal
from feature_extraction import extract_features
from model import AttentionLayer

from tqdm import tqdm


# ---------------- MODEL PATH ---------------- #
MODEL_PATH = "/home/sahith/projects/minor_project_408/outputs/models/best_model.keras"


# ---------------- CLASS NAMES ---------------- #
CLASS_NAMES = [
    "Healthy",
    "COPD",
    "Pneumonia"
]


# ---------------- PREPARE DATA ---------------- #
def prepare_data(df):

    mel_list = []
    hand_list = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        signal = row["signal"]

        label = row["label"]

        signal = preprocess_audio_from_signal(signal)

        mel, hand = extract_features(signal)

        # Normalize handcrafted features
        hand = (
            hand - np.mean(hand)
        ) / (
            np.std(hand) + 1e-6
        )

        mel = mel[..., np.newaxis]

        mel_list.append(mel)
        hand_list.append(hand)
        labels.append(label)

    return (
        np.array(mel_list),
        np.array(hand_list),
        np.array(labels)
    )


# ---------------- PATIENT SPLIT ---------------- #
def patient_split(df):

    patients = df["patient_id"].unique()

    train_p, temp_p = train_test_split(
        patients,
        test_size=0.30,
        random_state=42
    )

    val_p, test_p = train_test_split(
        temp_p,
        test_size=0.50,
        random_state=42
    )

    test_df = df[df["patient_id"].isin(test_p)]

    return test_df


# ---------------- MAIN ---------------- #
if __name__ == "__main__":

    print("\nLoading dataset...")

    df = load_icbhi_cycles()

    test_df = patient_split(df)

    print("\nPreparing test data...")

    X_test_mel, X_test_hand, y_test = prepare_data(test_df)

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "AttentionLayer": AttentionLayer
        },
        compile=False
    )

    print("\nPredicting...")

    predictions = model.predict(
        [X_test_mel, X_test_hand],
        batch_size=32
    )

    y_pred = np.argmax(predictions, axis=1)

    # ---------------- ACCURACY ---------------- #
    acc = accuracy_score(y_test, y_pred)

    print("\nTest Accuracy:")
    print(f"{acc * 100:.2f}%")

    # ---------------- CLASSIFICATION REPORT ---------------- #
    print("\nClassification Report:\n")

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )

    print(report)

    # ---------------- CONFUSION MATRIX ---------------- #
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:\n")

    print(cm)

    # ---------------- PLOT ---------------- #
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()