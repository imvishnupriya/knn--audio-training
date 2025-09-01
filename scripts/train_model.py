import os
import joblib
import time
from tqdm import tqdm
import threading
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.audio_utils import load_clip_and_extract_features, feature_index_table

# ========================
# Directories and Files
# ========================
data_directory = 'data/dataset_clips'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

FEATURES_FILE = 'X_features.npy'
LABELS_FILE = 'y_labels.npy'

# ========================
# Data Gathering
# ========================
def gather_files_and_labels(data_directory):
    filepaths, labels = [], []
    for folder_name in os.listdir(data_directory):
        folder_path = os.path.join(data_directory, folder_name)
        if os.path.isdir(folder_path):
            label = folder_name.split('_')[-1] if '_' in folder_name else folder_name
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    filepaths.append(os.path.join(folder_path, file))
                    labels.append(label)
    return filepaths, labels

def extract_features_parallel(filepaths, labels, max_workers=8):
    X, y = [], []
    file_count = [0]
    lock = threading.Lock()

    def process(filepath, label):
        features = load_clip_and_extract_features(filepath)
        with lock:
            file_count[0] += 1
            print(f"\rProcessed files: {file_count[0]} / {len(filepaths)}", end='', flush=True)
        return features, label

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process, fp, lbl) for fp, lbl in zip(filepaths, labels)]
        for future in as_completed(futures):
            try:
                features, label = future.result()
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error processing file: {e}")
    return np.array(X), np.array(y)

# ========================
# Load or Extract Features
# ========================
if os.path.exists(FEATURES_FILE) and os.path.exists(LABELS_FILE):
    print("Loading cached features...")
    X = np.load(FEATURES_FILE)
    y = np.load(LABELS_FILE)
    df = feature_index_table(63)
    print(df)
else:
    print("Extracting features, please wait...")
    filepaths, labels = gather_files_and_labels(data_directory)
    print(f"Total files: {len(filepaths)}")
    X, y = extract_features_parallel(filepaths, labels, max_workers=8)
    np.save(FEATURES_FILE, X)
    np.save(LABELS_FILE, y)
    print(f"Features cached to {FEATURES_FILE} and {LABELS_FILE}")
    df = feature_index_table(63)
    print(df)
    

# ========================
# Encode Labels
# ========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# ========================
# Train/Test Split
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# ========================
# Scale Features
# ========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

# ========================
# Define Models
# ========================
models = {
    "kNN": KNeighborsClassifier(n_neighbors=5, metric='manhattan', n_jobs=-1),
    "DT": DecisionTreeClassifier(criterion="gini", max_depth=20, min_samples_split=2, random_state=42),
    "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Ada-Boost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=20, min_samples_split=2),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    ),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale")
}

# ========================
# Training & Evaluation
# ========================
def show_progress(stop_event, model_name):
    with tqdm(total=0, position=0, bar_format="{desc} {elapsed}") as pbar:
        pbar.set_description(f"Training {model_name}")
        while not stop_event.is_set():
            time.sleep(1)
            pbar.update(0)

# Store per-class accuracy
per_class_results = {cls: {} for cls in class_names}
overall_acc = {}

for name, model in models.items():
    print(f"\n===== Training {name} =====")
    stop_event = threading.Event()
    progress_thread = threading.Thread(target=show_progress, args=(stop_event, name))
    progress_thread.start()

    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time

    stop_event.set()
    progress_thread.join()
    print(f"Finished {name} in {elapsed_time:.2f} seconds")

    # Save model
    joblib.dump(model, os.path.join(model_dir, f"{name.replace(' ', '_').lower()}_model.pkl"))

    # Overall Accuracy
    acc = accuracy_score(y_test, y_pred)
    overall_acc[name] = round(acc, 4)

    # Per-class accuracy
    for cls_idx, cls_name in enumerate(class_names):
        cls_mask = (y_test == cls_idx)
        cls_acc = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
        per_class_results[cls_name][name] = round(cls_acc, 4)

# ========================
# Create Per-Class Accuracy DataFrame
# ========================
df_per_class = pd.DataFrame(per_class_results).T
df_per_class.loc["Overall"] = overall_acc

print("\n=== Per-Class Accuracy Table ===")
print(df_per_class)
print(" ")
