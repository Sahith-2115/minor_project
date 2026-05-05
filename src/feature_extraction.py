import numpy as np
import librosa

SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 20


def extract_mel_spectrogram(signal):
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-6)

    return mel_spec_db


def spec_augment(mel):
    mel = mel.copy()

    # Time masking
    t = np.random.randint(10, 30)
    if mel.shape[1] > t:
        t0 = np.random.randint(0, mel.shape[1] - t)
        mel[:, t0:t0+t] = 0

    # Frequency masking
    f = np.random.randint(5, 15)
    if mel.shape[0] > f:
        f0 = np.random.randint(0, mel.shape[0] - f)
        mel[f0:f0+f, :] = 0

    return mel


def extract_handcrafted_features(signal):
    features = []

    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    zcr = librosa.feature.zero_crossing_rate(signal)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(y=signal, sr=SR)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=SR)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    chroma = librosa.feature.chroma_stft(y=signal, sr=SR)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    return np.array(features)


def extract_features(signal):
    mel = extract_mel_spectrogram(signal)
    hand = extract_handcrafted_features(signal)
    return mel, hand