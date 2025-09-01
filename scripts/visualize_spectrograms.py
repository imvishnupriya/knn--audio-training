import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from utils.feature_extraction import compute_stht

# ----------------------
# Configuration
# ----------------------
data_directory = 'data/dataset_clips'
selected_classes = ['chainsaw', 'forest', 'handsaw', 'speech', 'vehicle']
max_samples = 16000
amp_threshold = 0.001  # amplitude below this is considered silence
frame_size = 1024     # must match compute_stht

# ----------------------
# Classes
# ----------------------
all_classes = os.listdir(data_directory)
classes = sorted([cls for cls in all_classes if cls in selected_classes])
num_classes = len(classes)

plt.figure(figsize=(4 * num_classes, 8))  # 2 rows: amplitude + spectrogram

# ----------------------
# Loop through classes
# ----------------------
for i, label in enumerate(classes):
    class_path = os.path.join(data_directory, label)
    files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    if not files:
        continue
    file_path = os.path.join(class_path, files[1])
    signal, sr = librosa.load(file_path, sr=16000)

    # ----------------------
    # Trim silence
    # ----------------------
    non_silent_indices = np.where(np.abs(signal) > amp_threshold)[0]
    if len(non_silent_indices) == 0:
        continue  # skip completely silent
    start_idx = non_silent_indices[0]
    end_idx = non_silent_indices[-1]
    signal_trimmed = signal[start_idx:end_idx + 1]

    # ----------------------
    # Pad or truncate to max_samples
    # ----------------------
    if len(signal_trimmed) < max_samples:
        signal_plot = np.pad(signal_trimmed, (0, max_samples - len(signal_trimmed)), mode='constant')
    else:
        signal_plot = signal_trimmed[:max_samples]

    # Pad again if too short for STHT
    if len(signal_plot) < frame_size:
        signal_plot = np.pad(signal_plot, (0, frame_size - len(signal_plot)), mode='constant')

    # ----------------------
    # Amplitude vs Sample
    # ----------------------
    plt.subplot(2, num_classes, i + 1)
    plt.plot(signal_plot)
    plt.title(f'Amplitude - {label}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.xlim(0, max_samples)

    # ----------------------
    # STHT Spectrogram
    # ----------------------
    spectrogram = compute_stht(signal_plot)
    log_spec = np.log1p(np.abs(spectrogram))
    plt.subplot(2, num_classes, i + 1 + num_classes)
    plt.imshow(log_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f'STHT Spectrogram - {label}')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')

plt.tight_layout()
plt.show()
