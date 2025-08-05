import librosa
import soundfile as sf
import numpy as np
import random


def time_shift_with_rollover(file_path, output_path, max_shift_seconds=0.5, TARGET_SR=16000, BIT_DEPTH='PCM_16'):
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    shift_max = int(sr * max_shift_seconds)  # max shift in samples
    shift = random.randint(-shift_max, shift_max)  # randomly choose shift

    # Apply circular shift
    y_shifted = np.roll(y, shift)
    sf.write(output_path, y_shifted, sr, subtype=BIT_DEPTH)