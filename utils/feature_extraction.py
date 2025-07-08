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

def extract_llf(spectrogram, sr=16000):
    features = []
    freq_bins = np.arange(spectrogram.shape[0])
    prev_abs_frame = None
    for frame in tqdm(spectrogram.T, desc="Extracting LLFs", leave=False):
        abs_frame = np.abs(frame) + 1e-8
        norm_frame = abs_frame / np.sum(abs_frame)
        energy = np.sum(abs_frame ** 2)
        centroid = np.sum(freq_bins * abs_frame) / np.sum(abs_frame)
        spread = np.sqrt(np.sum(((freq_bins - centroid) ** 2) * abs_frame) / np.sum(abs_frame))
        skewness = (np.sum(((freq_bins - centroid) ** 3) * abs_frame) / np.sum(abs_frame)) / (spread ** 3 + 1e-8)
        kurtosis = (np.sum(((freq_bins - centroid) ** 4) * abs_frame) / np.sum(abs_frame)) / (spread ** 4 + 1e-8)
        flatness = np.exp(np.mean(np.log(abs_frame))) / (np.mean(abs_frame) + 1e-8)
        entropy = -np.sum(norm_frame * np.log2(norm_frame))
        slope = np.polyfit(freq_bins, abs_frame, 1)[0]
        rolloff = np.argmax(np.cumsum(abs_frame) >= 0.85 * np.sum(abs_frame))
        zero_crossings = np.count_nonzero(np.diff(np.sign(frame)))
        rms = np.sqrt(np.mean(abs_frame ** 2))
        crest_factor = np.max(abs_frame) / (rms + 1e-8)
        sorted_spectrum = np.sort(abs_frame)
        high_energy = np.mean(sorted_spectrum[int(0.9 * len(sorted_spectrum)):])
        low_energy = np.mean(sorted_spectrum[:int(0.1 * len(sorted_spectrum))])
        contrast = high_energy - low_energy
        bandwidth = np.sqrt(np.sum(((freq_bins - centroid) ** 2) * abs_frame) / np.sum(abs_frame))
        decrease = np.sum((abs_frame[1:] - abs_frame[0]) / np.arange(1, len(abs_frame))) / (np.sum(abs_frame[1:]) + 1e-8)
        flux = 0.0 if prev_abs_frame is None else np.sqrt(np.sum((abs_frame - prev_abs_frame) ** 2)) / len(abs_frame)
        prev_abs_frame = abs_frame.copy()
        features.append([
            energy, centroid, spread, skewness, kurtosis,
            flatness, entropy, slope, rolloff, zero_crossings,
            rms, crest_factor, contrast, bandwidth, decrease, flux
        ])
    return np.array(features).flatten()