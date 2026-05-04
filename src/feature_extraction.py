import numpy as np
import librosa

# ---------------- CONFIG ---------------- #
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 20


# ---------------- MEL-SPECTROGRAM ---------------- #
def extract_mel_spectrogram(signal):
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize (z-score)
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-6)

    return mel_spec_db  # shape: (128, time)


# ---------------- HANDCRAFTED FEATURES ---------------- #
def extract_handcrafted_features(signal):
    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(signal)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=signal, sr=SR)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=SR)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=signal, sr=SR)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    return np.array(features)  # shape: (70,)


# ---------------- FULL PIPELINE ---------------- #
def extract_features(signal):
    mel = extract_mel_spectrogram(signal)
    handcrafted = extract_handcrafted_features(signal)

    return mel, handcrafted


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    from data_loader import load_icbhi_dataset
    from preprocessing import preprocess_audio

    df = load_icbhi_dataset()

    print("\nTesting feature extraction...\n")

    for i in range(3):
        path = df.iloc[i]["file_path"]

        signal = preprocess_audio(path)

        mel, hand = extract_features(signal)

        print(f"Sample {i+1}")
        print(f"Mel shape: {mel.shape}")
        print(f"Handcrafted shape: {hand.shape}")
        print("-" * 40)