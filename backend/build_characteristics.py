import os
import csv
import json
import random
import numpy as np
from ultralytics import YOLO
import cv2
import importlib

import sys
sys.path.append(os.path.join(os.getcwd(), "backend"))
app = importlib.import_module("app")

DATASET = os.environ.get("DATASET", os.path.join(os.getcwd(), "data", "Dataset project.v6i.multiclass", "valid"))
CLASSES_CSV = os.path.join(DATASET, "_classes.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.getcwd(), "models"))

rf = {}
with open(CLASSES_CSV, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    classes = header[1:]
    rows = list(reader)

items = []
for row in rows:
    filename = row[0]
    vec = [int(float(x)) if x.strip() != '' else 0 for x in row[1:]]
    idx = int(np.argmax(vec))
    items.append((os.path.join(DATASET, filename), classes[idx]))

by_cls = {}
for p, name in items:
    by_cls.setdefault(name, []).append(p)

yolo = YOLO(os.path.join(MODEL_PATH, "best.pt"))
random.seed(7)

def measure(path):
    img = cv2.imread(path)
    if img is None:
        return None
    det = yolo.predict(source=path, conf=0.15, imgsz=960, verbose=False)
    boxes = det[0].boxes
    roi = img
    if len(boxes) > 0:
        b = boxes[0]
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
        roi = img[y1:y2, x1:x2]
    m = app.extract_morphology(roi, scale_factor=(app.MM_PER_PX if app.MORPH_UNITS == "mm" else 1.0))
    return {
        "panjang_daun_mm": float(m["panjang_mm"]),
        "lebar_daun_mm": float(m["lebar_mm"]),
        "keliling_daun_mm": float(m["keliling_mm"]),
        "panjang_tulang_daun_mm": float(m["panjang_tulang_mm"]),
        "rasio_bentuk_daun": float(m["rasio_bentuk"]),
    }

out = {}
for name, paths in by_cls.items():
    random.shuffle(paths)
    sample = paths[:50]
    vals = {"panjang_daun_mm": [], "lebar_daun_mm": [], "keliling_daun_mm": [], "panjang_tulang_daun_mm": [], "rasio_bentuk_daun": []}
    for p in sample:
        r = measure(p)
        if not r:
            continue
        for k in vals.keys():
            vals[k].append(r[k])
    stats = {}
    for k, arr in vals.items():
        if not arr:
            continue
        a = np.array(arr, dtype=np.float32)
        stats[k] = {
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a) + 1e-6),
        }
    out[name] = stats

os.makedirs(MODEL_PATH, exist_ok=True)
with open(os.path.join(MODEL_PATH, "variety_characteristics.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("Saved:", os.path.join(MODEL_PATH, "variety_characteristics.json"))