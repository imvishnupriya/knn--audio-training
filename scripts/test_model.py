import os
import joblib
import pandas as pd
from collections import Counter
from tqdm import tqdm
from utils.feature_extraction import compute_stht, extract_llf
import librosa
import numpy as np

# ========================
# Load models and preprocessing
# ========================
model_dir = 'models'
model_files = ['knn_model.pkl', 'dt_model.pkl', 'rf_model.pkl', 'ada-boost_model.pkl', 'svm_model.pkl']

models = {}
for file in model_files:
    name = os.path.splitext(file)[0]
    models[name] = joblib.load(os.path.join(model_dir, file))

scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))

# ========================
# Predict function
# ========================
def predict_new_audio_majority(audio_path, clip_length=1.0, sr=16000, cache_features=True):
    cache_file = f"{os.path.splitext(os.path.basename(audio_path))[0]}_features.npy"
    if cache_features and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        X_test = np.load(cache_file)
    else:
        signal, _ = librosa.load(audio_path, sr=sr, mono=True)
        clip_samples = int(clip_length * sr)
        num_clips = len(signal) // clip_samples
        print(f"Total clips: {num_clips}")

        features_list = []
        for i in tqdm(range(num_clips), desc="Extracting Features", unit="clip"):
            start = i * clip_samples
            end = start + clip_samples
            clip = signal[start:end]
            spectrogram = compute_stht(clip)
            features = extract_llf(spectrogram)
            features_list.append(features)

        X_test = np.array(features_list)
        if cache_features:
            np.save(cache_file, X_test)
            print(f"Features cached to {cache_file}")

    print("Scaling features...")
    X_test_scaled = scaler.transform(X_test)

    # ========================
    # Get predictions from all models
    # ========================
    all_model_preds = []
    for name, model in models.items():
        preds = model.predict(X_test_scaled)
        all_model_preds.append(preds)

    all_model_preds = np.array(all_model_preds)  # shape: (num_models, num_clips)

    # ========================
    # Majority vote per clip
    # ========================
    final_preds_indices = []
    for clip_idx in range(X_test_scaled.shape[0]):
        clip_preds = all_model_preds[:, clip_idx]
        vote = Counter(clip_preds).most_common(1)[0][0]
        final_preds_indices.append(vote)

    final_preds_labels = label_encoder.inverse_transform(final_preds_indices)
    return final_preds_labels

# ========================
# Main
# ========================
if __name__ == '__main__':
    new_audio_path = 'data/test_audio/test.mp3'
    print(f"\nTesting new audio: {new_audio_path}")
    predictions = predict_new_audio_majority(new_audio_path)

    counts = Counter(predictions)
    majority_vote = counts.most_common(1)[0][0]
    print(f"\nMajority vote (FINAL PREDICTION): {majority_vote}")
    print(" ")
