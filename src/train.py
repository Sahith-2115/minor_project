import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

from data_loader import load_icbhi_cycles
from preprocessing import preprocess_audio_from_signal
from feature_extraction import extract_features, spec_augment
from model import build_model
from augmentation import augment_audio


MODEL_SAVE_PATH = "../outputs/models/best_model.keras"


# 🔥 FOCAL LOSS
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.sum(weight * cross_entropy, axis=1)
    return loss


def prepare_data(df, is_training=False):
    mel_list, hand_list, labels = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        signal = row["signal"]
        label = row["label"]

        signal = preprocess_audio_from_signal(signal)

        if is_training and np.random.rand() < 0.5:
            signal = augment_audio(signal)
            signal = preprocess_audio_from_signal(signal)

        mel, hand = extract_features(signal)

        # SpecAugment
        if is_training and np.random.rand() < 0.3:
            mel = spec_augment(mel)

        # Normalize handcrafted
        hand = (hand - np.mean(hand)) / (np.std(hand) + 1e-6)

        mel = mel[..., np.newaxis]

        mel_list.append(mel)
        hand_list.append(hand)
        labels.append(label)

    return np.array(mel_list), np.array(hand_list), np.array(labels)


def patient_split(df):
    patients = df["patient_id"].unique()

    train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
    val_p, test_p = train_test_split(temp_p, test_size=0.50, random_state=42)

    return (
        df[df["patient_id"].isin(train_p)],
        df[df["patient_id"].isin(val_p)],
        df[df["patient_id"].isin(test_p)]
    )


if __name__ == "__main__":

    df = load_icbhi_cycles()

    train_df, val_df, test_df = patient_split(df)

    X_train_mel, X_train_hand, y_train = prepare_data(train_df, True)
    X_val_mel, X_val_hand, y_val = prepare_data(val_df, False)

    y_train = tf.keras.utils.to_categorical(y_train, 4)
    y_val = tf.keras.utils.to_categorical(y_val, 4)

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=focal_loss(),
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=4)
    ]

    history = model.fit(
        [X_train_mel, X_train_hand], y_train,
        validation_data=([X_val_mel, X_val_hand], y_val),
        epochs=80,
        batch_size=32,
        callbacks=callbacks
    )

    print("\nTraining complete.")