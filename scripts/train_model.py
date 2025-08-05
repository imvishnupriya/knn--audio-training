import os
import joblib
import threading
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.audio_utils import load_clip_and_extract_features

data_directory = 'data/dataset_clips'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

FEATURES_FILE = 'X_features.npy'
LABELS_FILE = 'y_labels.npy'

def gather_files_and_labels(data_directory):
    filepaths, labels = [], []
    for folder_name in os.listdir(data_directory):
        folder_path = os.path.join(data_directory, folder_name)
        if os.path.isdir(folder_path):
            # Extract label from folder name (e.g. tr_chainsaw → chainsaw)
            if '_' in folder_name:
                label = folder_name.split('_')[-1]
            else:
                label = folder_name

            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    filepaths.append(os.path.join(folder_path, file))
                    labels.append(label)
    return filepaths, labels



def extract_features_parallel(filepaths, labels, max_workers=8):
    X, y = [], []
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


# MAIN EXECUTION 
if os.path.exists(FEATURES_FILE) and os.path.exists(LABELS_FILE):
    print("Loading cached features...")
    X = np.load(FEATURES_FILE)
    y = np.load(LABELS_FILE)
else:
    print("Extracting features, please wait...")
    filepaths, labels = gather_files_and_labels(data_directory)
    print(f"Total files: {len(filepaths)}")
    X, y = extract_features_parallel(filepaths, labels, max_workers=8)
    np.save(FEATURES_FILE, X)
    np.save(LABELS_FILE, y)
    print(f"Features cached to {FEATURES_FILE} and {LABELS_FILE}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

print("Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', n_jobs=-1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("\nClassification Report:")
print(classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred)))

joblib.dump(knn, os.path.join(model_dir, 'knn_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

print(f"\n✅ Model, scaler, and label encoder saved to '{model_dir}'.")
