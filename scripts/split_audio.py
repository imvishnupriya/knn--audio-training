import os
from utils.audio_utils import split_audio_to_clips

# Hardcoded usage consistent with your earlier `index.py` workflow
raw_audio_dir = 'data/raw_audio'
data_directory = 'data/dataset_clips'
audio_files = {
    'chainsaw': 'chainsaw_long.mp3',
    'handsaw' : 'handsaw_long.mp3',
    'vehicle': 'vehicle_long.mp3',
    'speech': 'speech_long.mp3',
    'forest': 'forest_long.mp3'
}

for label, filename in audio_files.items():
    output_dir = os.path.join(data_directory, label)
    input_path = os.path.join(raw_audio_dir, filename)

    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        if not os.path.exists(input_path):
            print(f"Missing file: {input_path}. Please add it to proceed.")
            continue
        print(f"Creating clips for {label} from {input_path}...")
        split_audio_to_clips(input_path, output_dir, label)
