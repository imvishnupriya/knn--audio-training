# 🌳 IASN KNN Acoustic Event Detection Pipeline

A real-time monitoring system for illegal logging and environmental acoustic events using intelligent acoustic sensor nodes (IASN). Uses low-level feature extraction (LLFs) with Short-Time Hartley Transform (STHT) and classifies acoustic events using a K-Nearest Neighbors (KNN) classifier.

---

## 📂 Project Structure
```bash
knn-audio-training/
│
├── index.py # Main orchestrator: runs the entire pipeline
│
├── scripts/
│ ├── init.py # Package marker (empty)
│ ├── split_audio.py # Splits long raw audios into 1s clips for training
│ ├── visualize_spectrograms.py # Visualizes STHT spectrograms for each class
│ ├── train_model.py # Trains and saves the KNN model
│ └── test_model.py # Tests new audio using the trained model
│
├── utils/
│ ├── init.py # Package marker (empty)
│ ├── audio_utils.py # Splitting, loading clips, etc.
│ └── feature_extraction.py # STHT and LLF extraction
│
├── data/
│ ├── raw_audio/ # Place raw long .mp3/.wav recordings here
│ ├── dataset_clips/ # Auto-generated 1s clips for training
│ └── test_audio/ # Place test audios for classification here
│
├── models/ # Saved KNN model, scaler, and label encoder
│
├── requirements.txt # Required Python libraries
└── README.md # Project overview and instructions
```
---

## 🎯 Project Goals

- Detect illegal logging activities using acoustic features.
- Differentiate between forest sounds, vehicle noise, speech, and logging events.
- Enable real-time, low-power edge deployment using lightweight feature extraction and KNN.
- Provide visual tools for dataset sanity checking and result analysis.

---

## ⚙️ Installation

✅ Ensure Python 3.8+ is installed.

✅ Install dependencies:
```bash
pip install -r requirements.txt
🚀 Pipeline Execution
Run the full pipeline:

python index.py
This will:

Split long raw audios into clips.

Visualize STHT spectrograms for each class.

Train the KNN model and save to models/.

Test new audio in data/test_audio/ and display per-clip + majority predictions.

Manual Step-by-Step:
Split raw audio:


python -m scripts.split_audio
Visualize spectrograms:


python -m scripts.visualize_spectrograms
Train the model:


python -m scripts.train_model
Test on new field data:


python -m scripts.test_model
🎨 Features
STHT-based spectrogram generation.

16 LLFs per frame (energy, centroid, bandwidth, entropy, etc.).

Visual data inspection.

KNN classification with per-clip prediction and majority voting.

tqdm progress bars for transparency.

Modular, scalable structure ready for CNN/LSTM if needed.

📈 Potential Extensions
Batch test multiple field recordings automatically.

Log classification results to CSV.

Add confusion matrices for analysis.

ONNX/TFLite export for microcontroller deployment.

Real-time edge streaming and alert integration.

🤝 Contributing
Contributions welcome for enhancements such as batch testing, cloud integration, or model improvements.

