from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("yolov8n.pt")  # Gunakan yolov8s.pt untuk hasil lebih baik

# Training parameters
results = model.train(
    data=data_yaml_path,      # File konfigurasi dataset
    epochs=50,                # Jumlah epoch
    imgsz=640,                # Ukuran gambar
    batch=16,                 # Batch size
    name="chili_leaf_v5",     # Nama training session
    patience=10,              # Early stopping
    save=True,                # Save model
    device=0,                 # Gunakan GPU
    verbose=True,             # Tampilkan progress
    # Data augmentation
    degrees=45,               # Rotasi -45° sampai +45°
    flipud=0.5,               # Flip vertikal
    fliplr=0.5                # Flip horizontal
)

print("✅ Training selesai!")