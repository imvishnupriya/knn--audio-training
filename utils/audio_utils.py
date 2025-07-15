import os
import librosa
import soundfile as sf
from tqdm import tqdm
from .feature_extraction import compute_stht, extract_llf
import numpy as np

def split_audio_to_clips(audio_path, output_dir, label, clip_length=1.0, sr=16000):
    signal, _ = librosa.load(audio_path, sr=sr)
    clip_samples = int(clip_length * sr)
    num_clips = len(signal) // clip_samples
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_clips), desc=f"Splitting '{label}'"):
        start = i * clip_samples
        end = start + clip_samples
        clip = signal[start:end]
        filename = os.path.join(output_dir, f'{label}_{i}.wav')
        sf.write(filename, clip, sr)

# def load_clips_from_folder(data_dir):
#     X, y = [], []
#     for label in tqdm(os.listdir(data_dir), desc="Loading folders"):
#         label_dir = os.path.join(data_dir, label)
#         if not os.path.isdir(label_dir):
#             continue
#         files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
#         for file in tqdm(files, desc=f"Processing '{label}'", leave=False):
#             path = os.path.join(label_dir, file)
#             signal, sr = librosa.load(path, sr=16000)
#             spectrogram = compute_stht(signal)
#             features = extract_llf(spectrogram)
#             X.append(features)
#             y.append(label)
#     return np.array(X), np.array(y)


def load_clip_and_extract_features(filepath, sr=16000):
    signal, _ = librosa.load(filepath, sr=sr, mono=True)
    spectrogram = compute_stht(signal)
    features = extract_llf(spectrogram)
    return np.array(features)
