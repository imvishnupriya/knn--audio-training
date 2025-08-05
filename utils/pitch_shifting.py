import librosa
import soundfile as sf

def pitch_shift_audio(file_path, output_path, n_steps, TARGET_SR=16000, BIT_DEPTH='PCM_16'):
    y, sr = librosa.load(file_path, sr=TARGET_SR)  # force resample to 16kHz
    y_shifted = librosa.effects.pitch_shift(y, sr=TARGET_SR, n_steps=n_steps)
    sf.write(output_path, y_shifted, TARGET_SR, subtype=BIT_DEPTH)