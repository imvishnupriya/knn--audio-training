import subprocess
import sys

def run_script(module_path):
    print(f"\nRunning {module_path} ...\n" + "="*140)
    process = subprocess.Popen(
        [sys.executable, "-m", module_path],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    process.communicate()
    print("="*140)

if __name__ == '__main__':
    run_script("scripts.split_audio")
    run_script("scripts.train_model")
    run_script("scripts.test_model")
    run_script("scripts.visualize_spectrograms")
    print("\nEntire IASN KNN pipeline completed successfully!")
