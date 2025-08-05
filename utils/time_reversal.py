import librosa
import soundfile as sf


def reverse_audio(file_path, output_path, TARGET_SR=16000, BIT_DEPTH='PCM_16'):
    y, sr = librosa.load(file_path, sr=TARGET_SR)  # load and resample
    y_reversed = y[::-1]  # reverse the waveform
    sf.write(output_path, y_reversed, TARGET_SR, subtype=BIT_DEPTH)