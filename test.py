import os
import torch
import urllib.request
from ultralytics import YOLO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
det_path = os.path.join(MODEL_DIR, "yolov11n.pt")
seg_path = os.path.join(MODEL_DIR, "yolov11n-seg.pt")

# Download if missing
if not os.path.exists(det_path):
    print("Downloading yolov11n.pt...")
    urllib.request.urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11n.pt",
        det_path
    )

if not os.path.exists(seg_path):
    print("Downloading yolov11n-seg.pt...")
    urllib.request.urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11n-seg.pt",
        seg_path
    )


# Load models
det_model = YOLO(det_path)
seg_model = YOLO(seg_path)

SOURCE = "test.jpg"  # or '0' for webcam, or video path

# Detection inference
det_results = det_model(SOURCE, device=DEVICE)
det_results[0].save(save_dir="runs/detect/")

# Segmentation inference
seg_results = seg_model(SOURCE, device=DEVICE)
seg_results[0].save(save_dir="runs/segment/")

print("Inference done. Results saved in runs/ folder.")
