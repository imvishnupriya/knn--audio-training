import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from utils.audio_utils import load_clips_from_folder

data_directory = 'data/dataset_clips'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

print("Loading and extracting features from dataset...")
X, y = load_clips_from_folder(data_directory)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

print("Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("\nClassification Report:")
print(classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred)))

joblib.dump(knn, os.path.join(model_dir, 'knn_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

print(f"\nModel, scaler, and label encoder saved to '{model_dir}'.")