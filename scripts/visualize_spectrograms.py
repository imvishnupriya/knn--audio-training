import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from utils.feature_extraction import compute_stht

# Hardcoded visualization consistent with your earlier workflow
data_directory = 'data/dataset_clips'
# classes = sorted(os.listdir(data_directory))

all_classes = os.listdir(data_directory)
classes = sorted([cls for cls in all_classes if cls in ['chainsaw', 'forest', 'handsaw', 'speech', 'vehicle']])
num_classes = len(classes)

plt.figure(figsize=(4 * num_classes, 5))

for i, label in enumerate(classes):
    class_path = os.path.join(data_directory, label)
    files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    if not files:
        continue
    file_path = os.path.join(class_path, files[0])
    signal, sr = librosa.load(file_path, sr=16000)

    spectrogram = compute_stht(signal)
    log_spec = np.log1p(np.abs(spectrogram))
    time_axis = range(log_spec.shape[1])

    plt.subplot(1, num_classes, i + 1)
    plt.imshow(log_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f'STHT Spectrogram - {label}')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')

plt.tight_layout()
plt.show()
