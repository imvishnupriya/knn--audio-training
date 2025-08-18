import subprocess
import sys
import os
import time

def run_script(module_path):
    print(f"\nRunning {module_path} ...\n" + "="*140)
    start = time.time()
    process = subprocess.Popen(
        [sys.executable, "-m", module_path],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    process.communicate()
    end = time.time()
    print(f"{module_path} completed in {end - start:.2f} seconds")
    print("="*140)

def remove_cache_files():
    cache_files = ["X_features.npy", "y_labels.npy", "test_features.npy"]
    removed = False
    for file in cache_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed cache file: {file}")
            removed = True
    if not removed:
        print("No cache files found to remove.")

if __name__ == '__main__':
    user_input = input("Do you want to remove cached feature files before running? (y/n): ").strip().lower()
    if user_input == 'y':
        print("\nRemoving cache files...")
        remove_cache_files()
    else:
        print("Skipping cache removal, using existing feature caches if available.")

    pipeline_start = time.time()

    run_script("scripts.split_audio")
    run_script("scripts.audio_aug")

    run_script("scripts.train_model")
    run_script("scripts.test_model")
    run_script("scripts.visualize_spectrograms")

    pipeline_end = time.time()
    print(f"\nEntire IASN KNN pipeline completed successfully in {pipeline_end - pipeline_start:.2f} seconds!")
