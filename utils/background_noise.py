import librosa
import soundfile as sf
import numpy as np
import os
import random

def add_noise(input_path, output_path, forest_folder, TARGET_SR=16000, BIT_DEPTH='PCM_16'):
    # Load original signal
    x, sr = librosa.load(input_path, sr=TARGET_SR)

    # Pick a random forest background
    forest_files = [f for f in os.listdir(forest_folder) if f.endswith('.wav') or f.endswith('.mp3')]
    forest_path = os.path.join(forest_folder, random.choice(forest_files))
    g, _ = librosa.load(forest_path, sr=TARGET_SR)

    # Match length: loop or trim forest noise
    if len(g) < len(x):
        g = np.tile(g, int(np.ceil(len(x) / len(g))))
    g = g[:len(x)]

    # Mix with random weight
    w = random.uniform(0.1, 0.5)
    a = (1 - w) * x + w * g

    # Normalize to avoid clipping
    a = a / np.max(np.abs(a)) * 0.99

    # Save output
    sf.write(output_path, a, TARGET_SR, subtype=BIT_DEPTH)
