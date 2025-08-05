import librosa
import soundfile as sf
import numpy as np


def polarity_inversion(file_path, output_path, TARGET_SR = 16000, BIT_DEPTH = 'PCM_16'):
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    y_inverted = -1 * y  # Polarity inversion
    sf.write(output_path, y_inverted, sr, subtype=BIT_DEPTH)