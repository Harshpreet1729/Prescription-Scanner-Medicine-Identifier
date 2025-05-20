
import os
import subprocess

MODEL_PATH = "ocr_cnn_model.pth"
ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("🔄 Model not found. Training model first...")
    subprocess.run(["python", "train_model.py"], check=True)
else:
    print("✅ Model and encoder found. Skipping training.")

print("🚀 Launching GUI app...")
subprocess.run(["python", "gui_app.py"], check=True)