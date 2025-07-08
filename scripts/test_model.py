import os
import joblib
import pandas as pd
from collections import Counter
from tqdm import tqdm
from utils.feature_extraction import compute_stht, extract_llf
import librosa

model_dir = 'models'
knn = joblib.load(os.path.join(model_dir, 'knn_model.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))

def predict_new_audio(audio_path, clip_length=1.0, sr=16000):
    signal, _ = librosa.load(audio_path, sr=sr)
    clip_samples = int(clip_length * sr)
    num_clips = len(signal) // clip_samples
    predictions = []
    for i in tqdm(range(num_clips), desc="🔍 Processing Clips", unit="clip"):
        start = i * clip_samples
        end = start + clip_samples
        clip = signal[start:end]
        spectrogram = compute_stht(clip)
        features = extract_llf(spectrogram)
        features_scaled = scaler.transform([features])
        pred = knn.predict(features_scaled)
        pred_label = label_encoder.inverse_transform(pred)[0]
        predictions.append(pred_label)
    return predictions

if __name__ == '__main__':
    new_audio_path = 'data/test_audio/test.mp3'
    print(f"\n🔍 Testing new audio: {new_audio_path}")
    predictions = predict_new_audio(new_audio_path)
    df = pd.DataFrame({"Clip Number": range(1, len(predictions) + 1), "Predicted Label": predictions})
    print(df.to_string(index=False))
    counts = Counter(predictions)
    majority_vote = counts.most_common(1)[0][0]
    print(f"\nMajority vote (final predicted region): {majority_vote}")