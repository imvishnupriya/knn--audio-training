import os
from tqdm import tqdm   # <-- add this
from utils.pitch_shifting import pitch_shift_audio
from utils.time_reversal import reverse_audio
from utils.fwd_bwd_shifting import time_shift_with_rollover
from utils.polarity_inv import polarity_inversion
from utils.background_noise import add_noise

# Input audio categories
input_folder_info = ["chainsaw", "forest", "handsaw", "speech", "vehicle"]
# Augmentation operations
operation = ['ps', 'tr', 'fbs', 'pi', 'bn']

# Constants
TARGET_SR = 16000
BIT_DEPTH = 'PCM_16'

for info in input_folder_info:
    input_folder = os.path.join('data/dataset_clips', info)
    
    for op in operation:
        output_folder = os.path.join('data/dataset_clips', f'{op}_{info}')
        os.makedirs(output_folder, exist_ok=True)
       
        if op == 'bn' and info != 'forest':
            continue

        # Wrap the file loop with tqdm
        for filename in tqdm(os.listdir(input_folder), desc=f"{op} on {info}", unit="file"):
            input_path = os.path.join(input_folder, filename)
            output_filename = f'{op}_{filename}'
            output_path = os.path.join(output_folder, output_filename)

            try:
                if op == 'ps':
                    pitch_shift_audio(input_path, output_path, n_steps=2, TARGET_SR=TARGET_SR, BIT_DEPTH=BIT_DEPTH)

                elif op == 'tr':
                    reverse_audio(input_path, output_path, TARGET_SR=TARGET_SR, BIT_DEPTH=BIT_DEPTH)

                elif op == 'fbs':
                    time_shift_with_rollover(input_path, output_path, TARGET_SR=TARGET_SR, BIT_DEPTH=BIT_DEPTH)

                elif op == 'pi':
                    polarity_inversion(input_path, output_path, TARGET_SR=TARGET_SR, BIT_DEPTH=BIT_DEPTH)

                elif op == 'bn':
                    forest_folder = 'data/dataset_clips/forest'
                    add_noise(input_path, output_path, forest_folder, TARGET_SR=TARGET_SR, BIT_DEPTH=BIT_DEPTH)

            except Exception as e:
                tqdm.write(f"Error processing {filename} with operation {op}: {e}")  # keeps errors visible
