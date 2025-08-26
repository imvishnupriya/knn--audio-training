import numpy as np
from tqdm import tqdm


def dht(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    cas = np.cos(2 * np.pi * k * n / N) + np.sin(2 * np.pi * k * n / N)
    return np.dot(cas, x)

def compute_stht(signal, frame_size=512, hop_size=256):
    window = np.hamming(frame_size)
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    spectrogram = np.zeros((frame_size, num_frames))
    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size] * window
        spectrogram[:, i] = dht(frame)
    return spectrogram

def extract_llf(spectrogram, sr=16000, n_subbands=11):
    """
    Extract 29 LLF features per frame from the STHT spectrogram.
    Returns a flattened feature vector (all frames concatenated).
    """
    features = []
    freq_bins = np.arange(spectrogram.shape[0])
    prev_abs_frame = None
    
    for frame in tqdm(spectrogram.T, desc="Extracting LLFs", leave=False):
        abs_frame = np.abs(frame) + 1e-8
        norm_frame = abs_frame / np.sum(abs_frame)
        
        # --- Single-valued features ---
        energy = np.sum(abs_frame ** 2)                # total energy
        rms = np.sqrt(np.mean(abs_frame ** 2))         # RMS energy
        centroid = np.sum(freq_bins * abs_frame) / np.sum(abs_frame)
        std = np.std(abs_frame)

        spread = np.sqrt(np.sum(((freq_bins - centroid) ** 2) * abs_frame) / np.sum(abs_frame))
        skewness = (np.sum(((freq_bins - centroid) ** 3) * abs_frame) / np.sum(abs_frame)) / (spread ** 3 + 1e-8)
        kurtosis = (np.sum(((freq_bins - centroid) ** 4) * abs_frame) / np.sum(abs_frame)) / (spread ** 4 + 1e-8)
        flatness = np.exp(np.mean(np.log(abs_frame))) / (np.mean(abs_frame) + 1e-8)
        entropy = -np.sum(norm_frame * np.log2(norm_frame))
        slope = np.polyfit(freq_bins, abs_frame, 1)[0]
        crest_factor = np.max(abs_frame) / (rms + 1e-8)
        
        sorted_spectrum = np.sort(abs_frame)
        high_energy = np.mean(sorted_spectrum[int(0.9 * len(sorted_spectrum)):])
        low_energy = np.mean(sorted_spectrum[:int(0.1 * len(sorted_spectrum))])
        contrast = high_energy - low_energy
        centre = (high_energy + low_energy)/2

        bandwidth = np.sqrt(np.sum(((freq_bins - centroid) ** 2) * abs_frame) / np.sum(abs_frame))
        decrease = np.sum((abs_frame[1:] - abs_frame[0]) / np.arange(1, len(abs_frame))) / (np.sum(abs_frame[1:]) + 1e-8)
        
        flux = 0.0 if prev_abs_frame is None else np.sqrt(np.sum((abs_frame - prev_abs_frame) ** 2)) / len(abs_frame)
        prev_abs_frame = abs_frame.copy()
        
        # --- Spectral rolloff at two thresholds (85% & 95%) ---
        rolloff_85 = np.argmax(np.cumsum(abs_frame) >= 0.85 * np.sum(abs_frame))
        rolloff_95 = np.argmax(np.cumsum(abs_frame) >= 0.95 * np.sum(abs_frame))
        
        # --- OBSC (Octave Band Spectral Contrast, 11 subbands) ---
        band_edges = np.linspace(0, len(abs_frame), n_subbands+1, dtype=int)
        obsc_values = []
        for b in range(n_subbands):
            band = abs_frame[band_edges[b]:band_edges[b+1]]
            if len(band) > 0:
                obsc_values.append(np.max(band) - np.min(band))
            else:
                obsc_values.append(0.0)
        
        # --- Collect all features ---
        frame_features = [
            energy, centre, std, rms, centroid, spread, skewness, kurtosis,
            flatness, entropy, slope, crest_factor, contrast,
            bandwidth, decrease, flux, rolloff_85, rolloff_95
        ] + obsc_values  # adds 11 values
        
        features.append(frame_features)
    
    return np.array(features).flatten()



# Energy (1)
# RMS (1)
# Center (1)
# SD (1)
# Centroid (1)
# Spread (1)
# Skewness (1)
# Kurtosis (1)
# Flatness (1)
# Entropy (1)
# Slope (1)
# Crest factor (1)
# Contrast (1)
# Bandwidth (1)
# Decrease (1)
# Flux (1)

# Roll-off (2: 85% & 95%)

# 11 OBSC
# --------------------
# 4 single-valued features
# 2 roll-off thresholds
# 11 OBSC sub-bands
# 16 + 2 + 11 = 29 features per frame
# ---------------------



import pandas as pd

def generate_feature_index_table(num_frames=63):
    feature_names = [
        "Energy", "Centre", "Standard Deviation", "RMS", "Spectral Centroid", "Spectral Spread",
        "Spectral Skewness", "Spectral Kurtosis", "Spectral Flatness",
        "Entropy", "Spectral Slope", "Crest Factor", "Spectral Contrast",
        "Spectral Bandwidth", "Spectral Decrease", "Spectral Flux",
        "Spectral Rolloff (85%)", "Spectral Rolloff (95%)"
    ] + [f"OBSC Sub-band {i+1}" for i in range(11)]  # OBSC expands into 11 features
    
    table_data = []
    feature_len = num_frames  # each feature repeats per frame
    
    start_idx = 1
    for fname in feature_names:
        end_idx = start_idx + feature_len - 1
        table_data.append([f"{start_idx}-{end_idx}", fname])
        start_idx = end_idx + 1
    
    df = pd.DataFrame(table_data, columns=["Feature Index", "Feature Description"])
    return df
