import os
import io
import json
import base64
import logging
from typing import Dict, Any, List
from collections import deque
from dotenv import load_dotenv

import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import efficientnet_b0
from skimage.morphology import skeletonize
import joblib
import importlib
import csv
# Ensure we import the external 'xgboost' library, not local training script
xgb = importlib.import_module("xgboost")

# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Konfigurasi dasar
API_VERSION = "1.0.0"
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.getcwd(), "models"))
MM_PER_PX = float(os.environ.get("MM_PER_PX", "0.1"))
MORPH_UNITS = os.environ.get("MORPH_UNITS", "mm").strip().lower()  # 'mm' atau 'px'
USE_COMBINED = True
USE_EFF_ONLY = False
USE_GATING = True
GATE_MARGIN = 0.15
GATE_CONF = 0.85
MORPH_BETA = 0.0
BLEND_META_WEIGHT = 0.70
META_FIRST = False
EFF_BLEND_EPS = 0.02
MORPH_SUPPRESS = 0.0
PROBA_ALPHA = 1.50
REBUILD_VARIETY_STATS = False
CLASSIFY_FULL_IMAGE = True
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Artefak model
yolo_model = None
effnet = None
effnet_cls = None
scaler = None
scaler_combined = None
xgb_original = None
xgb_combined = None
calibrator = None
CLASS_NAMES: List[str] = []


def load_artifacts() -> None:
    global yolo_model, effnet, effnet_cls, scaler, scaler_combined, selector, xgb_original, xgb_combined, CLASS_NAMES
    logger.info(f"Model path: {MODEL_PATH}")
    # YOLO
    yolo_path = os.path.join(MODEL_PATH, "best.pt")
    yolo_model = YOLO(yolo_path)
    logger.info("✓ YOLO loaded")
    effnet_path = os.path.join(MODEL_PATH, "best_efficientnet_model.pth")
    effnet_cls = efficientnet_b0(weights=None)
    effnet_cls.eval()
    effnet = efficientnet_b0(weights=None)
    effnet.eval()
    if os.path.exists(effnet_path):
        sd = torch.load(effnet_path, map_location="cpu")
        # Filter out classifier head parameters to avoid size mismatch
        sd_features = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
        try:
            ik2 = effnet.load_state_dict(sd_features, strict=False)
            logger.info(f"✓ EfficientNet feature weights loaded (missing={len(getattr(ik2,'missing_keys',[]))}, unexpected={len(getattr(ik2,'unexpected_keys',[]))})")
        except Exception as e:
            logger.warning(f"⚠ Failed loading EfficientNet feature weights: {e}")
        try:
            effnet.classifier = torch.nn.Identity()
            logger.info("✓ EfficientNet feature extractor ready")
        except Exception:
            logger.warning("⚠ Failed to set EfficientNet classifier to Identity")
    else:
        logger.warning("⚠ EfficientNet weights file not found; using base architecture")
        try:
            effnet.classifier = torch.nn.Identity()
        except Exception:
            pass
    # Scalers
    scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
    scaler_combined_path = os.path.join(MODEL_PATH, "scaler_combined.pkl")
    selector_path = os.path.join(MODEL_PATH, "selector.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info("✓ scaler.pkl loaded")
    if os.path.exists(scaler_combined_path):
        scaler_combined = joblib.load(scaler_combined_path)
        logger.info("✓ scaler_combined.pkl loaded")
    if os.path.exists(selector_path):
        try:
            selector = joblib.load(selector_path)
            if hasattr(selector, 'get_support'):
                sup = selector.get_support()
                logger.info(f"✓ selector.pkl loaded (input={len(sup)}, selected={int(np.sum(sup))})")
            else:
                logger.info("✓ selector.pkl loaded")
        except Exception as e:
            selector = None
            logger.warning(f"⚠ Failed to load selector.pkl: {e}")
    # XGBoost models
    xgb_orig_path = os.path.join(MODEL_PATH, "xgboost_meta_model_final.json")
    xgb_comb_path = os.path.join(MODEL_PATH, "xgboost_meta_model_combined.json")
    calib_path = os.path.join(MODEL_PATH, "calibrator_isotonic.pkl")
    if os.path.exists(xgb_orig_path):
        xgb_original = xgb.XGBClassifier()
        xgb_original.load_model(xgb_orig_path)
        logger.info("✓ XGBoost original loaded")
    if os.path.exists(xgb_comb_path):
        xgb_combined = xgb.XGBClassifier()
        xgb_combined.load_model(xgb_comb_path)
        logger.info("✓ XGBoost combined loaded")
    disable_calib = False
    if not disable_calib and os.path.exists(calib_path):
        try:
            calib = joblib.load(calib_path)
            calibrator = calib
            logger.info("✓ Calibrator loaded")
        except Exception as e:
            logger.warning(f"⚠ Failed to load calibrator: {e}")
    # Class names (urut indeks sesuai pelatihan)
    try:
        # Prioritaskan class_names.json dari artefak training jika tersedia
        class_json = os.path.join(MODEL_PATH, "class_names.json")
        if os.path.exists(class_json):
            with open(class_json, "r", encoding="utf-8") as f:
                arr = json.load(f)
                if isinstance(arr, list) and len(arr) > 0:
                    CLASS_NAMES = [str(x) for x in arr]
                    logger.info(f"✓ CLASS_NAMES loaded from class_names.json ({len(CLASS_NAMES)} kelas)")
        if not CLASS_NAMES:
            class_file = os.environ.get("CLASS_NAMES_FILE")
            if not class_file:
                class_file = os.path.join(os.getcwd(), "data", "Dataset project.v6i.multiclass", "train", "_classes.csv")
            if os.path.exists(class_file):
                with open(class_file, "r", encoding="utf-8") as f:
                    header = f.readline().strip()
                    parts = [p.strip() for p in header.split(",")]
                    CLASS_NAMES = parts[1:]
                    logger.info(f"✓ CLASS_NAMES loaded from train/_classes.csv ({len(CLASS_NAMES)} kelas)")
            else:
                logger.warning("⚠ CLASS_NAMES file not found; using index-based names")
    except Exception as e:
        logger.warning(f"⚠ Failed to load CLASS_NAMES: {e}")
    # NOTE: Gunakan urutan kelas dari dataset (train/_classes.csv) agar konsisten dengan pelatihan.
    # Jika model menyimpan kelas numerik (0..12), mapping indeks → nama varietas dilakukan saat memilih hasil.
    # Setelah CLASS_NAMES diketahui, bangun classifier EfficientNet pelatihan dan muat bobot penuh
    try:
        num_classes = len(CLASS_NAMES) if CLASS_NAMES else 13
        in_features = effnet_cls.classifier[1].in_features
        effnet_cls.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(512, num_classes),
        )
        if os.path.exists(effnet_path):
            try:
                sd_full = torch.load(effnet_path, map_location="cpu")
                effnet_cls.load_state_dict(sd_full, strict=False)
                logger.info("✓ EfficientNet classifier weights loaded")
            except Exception as e:
                logger.warning(f"⚠ Failed to load EfficientNet classifier weights: {e}")
        effnet_cls.eval()
    except Exception as e:
        logger.warning(f"⚠ Failed to build/load EfficientNet classifier: {e}")

    try:
        vc_path = os.path.join(MODEL_PATH, "variety_characteristics.json")
        if REBUILD_VARIETY_STATS or not os.path.exists(vc_path):
            cand = None
            try:
                for fn in os.listdir(MODEL_PATH):
                    if fn.lower().startswith("hasil_morfologi_") and fn.lower().endswith(".csv"):
                        cand = os.path.join(MODEL_PATH, fn)
                        break
            except Exception:
                cand = None
            src = cand or os.environ.get("MORPH_CSV_PATH")
            if src and os.path.exists(src):
                agg = {}
                with open(src, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        nm = str(row.get("Varietas", "").strip())
                        if not nm:
                            continue
                        try:
                            p = float(row.get("Panjang Daun (mm)", "nan"))
                            l = float(row.get("Lebar Daun (mm)", "nan"))
                            k = float(row.get("Keliling Daun (mm)", "nan"))
                            t = float(row.get("Panjang Tulang Daun (mm)", "nan"))
                            r = float(row.get("Rasio Bentuk Daun", "nan"))
                        except Exception:
                            continue
                        a = agg.setdefault(nm, {"p": [], "l": [], "k": [], "t": [], "r": []})
                        if np.isfinite(p): a["p"].append(p)
                        if np.isfinite(l): a["l"].append(l)
                        if np.isfinite(k): a["k"].append(k)
                        if np.isfinite(t): a["t"].append(t)
                        if np.isfinite(r): a["r"].append(r)
                out = {}
                for nm, a in agg.items():
                    def st(arr):
                        if len(arr) == 0:
                            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
                        x = np.array(arr, dtype=np.float32)
                        sd = float(np.std(x)) if len(x) > 1 else 0.0
                        return {"min": float(np.min(x)), "max": float(np.max(x)), "mean": float(np.mean(x)), "std": sd}
                    out[nm] = {
                        "panjang_daun_mm": st(a["p"]),
                        "lebar_daun_mm": st(a["l"]),
                        "keliling_daun_mm": st(a["k"]),
                        "panjang_tulang_daun_mm": st(a["t"]),
                        "rasio_bentuk_daun": st(a["r"]),
                    }
                try:
                    with open(vc_path, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False)
                    logger.info("✓ Variety characteristics built from CSV")
                except Exception as e:
                    logger.warning(f"⚠ Failed writing variety characteristics: {e}")
    except Exception as e:
        logger.warning(f"⚠ Failed building variety characteristics: {e}")


def encode_image_preview(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def preprocess_effnet(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    use_kaggle = str(os.environ.get("KAGGLE_PREPROCESS", "0")).lower() in ("1","true","yes","y")
    if use_kaggle:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return t(img_rgb).unsqueeze(0)


def find_longest_skeleton_path(skeleton_img):
    """Find longest path in skeleton using BFS (training style)"""
    points = np.column_stack(np.where(skeleton_img > 0))
    if len(points) == 0:
        return 0.0
    
    def bfs(start_point):
        queue = deque([(start_point, 0)])
        visited = set()
        visited.add(tuple(start_point))
        farthest_point = start_point
        max_distance = 0
        
        while queue:
            (y, x), dist = queue.popleft()
            if dist > max_distance:
                max_distance = dist
                farthest_point = (y, x)
            
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton_img.shape[0] and 0 <= nx < skeleton_img.shape[1]:
                    if skeleton_img[ny, nx] > 0 and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append(((ny, nx), dist + 1))
        
        return farthest_point, max_distance
    
    # Find one endpoint by BFS from arbitrary point
    start_point = points[0]
    farthest_point, _ = bfs(start_point)
    
    # Find the actual longest path from this endpoint
    _, path_length = bfs(farthest_point)
    
    return float(path_length)

def extract_morphology(roi_bgr: np.ndarray, scale_factor: float = 0.1) -> Dict[str, float]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, thr_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 25, 20], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(thr_inv, mask_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measurement_quality = {"status": "ok", "issues": []}

    if not contours:
        h, w = mask.shape
        measurement_quality["status"] = "warn"
        measurement_quality["issues"].append("kontur_tidak_ditemukan")
        panjang_mm = float(max(h, w)) * scale_factor
        lebar_mm = float(min(h, w)) * scale_factor
        keliling_mm = float(2 * (h + w)) * scale_factor
        luas_mm2 = float(h * w) * (scale_factor ** 2)
        return {
            "panjang_mm": panjang_mm,
            "lebar_mm": lebar_mm,
            "keliling_mm": keliling_mm,
            "rasio_bentuk": float(min(h, w) / max(h, w) if max(h, w) > 0 else 0.0),
            "panjang_tulang_mm": 0.0,
            "luas_mm2": luas_mm2,
            "measurement_quality": measurement_quality,
        }

    c = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(c))
    if area_px < 200.0:
        measurement_quality["status"] = "warn"
        measurement_quality["issues"].append("kontur_kecil")
    peri = float(cv2.arcLength(c, True))

    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), ang = rect
    panjang = float(max(rw, rh))
    lebar = float(min(rw, rh))
    rasio = float(lebar / panjang) if panjang > 0 else 0.0

    skeleton = skeletonize((mask > 0).astype(np.uint8)).astype(np.uint8) * 255
    panjang_tulang_px = find_longest_skeleton_path(skeleton)
    if panjang_tulang_px < (panjang * 0.25):
        measurement_quality["status"] = "warn"
        measurement_quality["issues"].append("tulang_daun_tidak_jelas")
        panjang_tulang_px = float(panjang * 0.55)

    panjang_mm = panjang * scale_factor
    lebar_mm = lebar * scale_factor
    keliling_mm = peri * scale_factor
    panjang_tulang_mm = panjang_tulang_px * scale_factor
    luas_mm2 = area_px * (scale_factor ** 2)

    return {
        "panjang_mm": panjang_mm,
        "lebar_mm": lebar_mm,
        "keliling_mm": keliling_mm,
        "rasio_bentuk": rasio,
        "panjang_tulang_mm": panjang_tulang_mm,
        "luas_mm2": luas_mm2,
        "measurement_quality": measurement_quality,
    }


def get_variety_characteristics(name: str) -> Dict[str, Any]:
    path = os.path.join(MODEL_PATH, "variety_characteristics.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get(name, {"name": name})
        except Exception:
            pass
    return {"name": name}


def build_health_payload() -> Dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "models_path": MODEL_PATH,
        "pipeline": "combined" if USE_COMBINED else "original",
        "loaded": {
            "yolo": yolo_model is not None,
            "efficientnet": effnet is not None,
            "efficientnet_classifier": effnet_cls is not None,
            "scaler": scaler is not None,
            "scaler_combined": scaler_combined is not None,
            "xgboost_original": xgb_original is not None,
            "xgboost_combined": xgb_combined is not None,
        },
    }


@app.route("/api/health", methods=["GET"]) 
def api_health():
    return jsonify(build_health_payload())


@app.route("/predict", methods=["POST"]) 
def predict():
    file = None
    if request.files:
        for k in ["image", "file", "photo", "picture"]:
            if k in request.files and request.files[k].filename:
                file = request.files[k]
                break
    if file is None:
        return jsonify({"success": False, "error": "no_file"}), 400

    tmp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(tmp_path)

    try:
        ignore_yolo = str(os.environ.get("IGNORE_YOLO", "0")).lower() in ("1","true","yes","y")
        img = cv2.imread(tmp_path)
        dets: List[Dict[str, Any]] = []
        if not ignore_yolo:
            res = yolo_model.predict(source=tmp_path, conf=0.05, imgsz=960, verbose=False)
            boxes = res[0].boxes
            names = res[0].names if hasattr(res[0], "names") else {}
            for i in range(len(boxes)):
                b = boxes[i]
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                conf = float(b.conf[0].item())
                cls_id = int(b.cls[0].item())
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, f"class_{cls_id}"),
                    "confidence": conf,
                })
        if (ignore_yolo or len(dets) == 0) and img is not None:

            h, w = img.shape[:2]
            dets.append({
                "bbox": [0.0, 0.0, float(w-1), float(h-1)],
                "class_id": -1,
                "class_name": "leaf",
                "confidence": 0.0,
            })

        best = max(dets, key=lambda d: d["confidence"]) 
        img = cv2.imread(tmp_path)
        x1, y1, x2, y2 = map(int, best["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
        roi_morph = img[y1:y2, x1:x2]
        roi_eff = img if CLASSIFY_FULL_IMAGE and img is not None else roi_morph

        morph = extract_morphology(roi_morph, scale_factor=(MM_PER_PX if MORPH_UNITS == "mm" else 1.0))
        panjang_mm = morph["panjang_mm"]
        lebar_mm = morph["lebar_mm"]
        keliling_mm = morph["keliling_mm"]
        panjang_tulang_mm = morph["panjang_tulang_mm"]

        with torch.no_grad():
            rois = []
            rois.append(roi_eff)
            try:
                ycc = cv2.cvtColor(roi_eff, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycc)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                y2 = clahe.apply(y)
                eq = cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)
                rois.append(eq)
            except Exception:
                rois.append(roi_eff)
            try:
                g_low = np.clip(((roi_eff.astype(np.float32)/255.0) ** 0.8) * 255.0, 0, 255).astype(np.uint8)
                g_high = np.clip(((roi_eff.astype(np.float32)/255.0) ** 1.2) * 255.0, 0, 255).astype(np.uint8)
                rois.append(g_low)
                rois.append(g_high)
            except Exception:
                pass
            try:
                rois.append(cv2.flip(roi_eff, 1))
            except Exception:
                pass
            try:
                h, w = roi_eff.shape[:2]
                c = (w/2, h/2)
                for ang in (-10, 10):
                    M = cv2.getRotationMatrix2D(c, ang, 1.0)
                    rot = cv2.warpAffine(roi_eff, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    rois.append(rot)
            except Exception:
                pass
            f_list = []
            p_list = []
            for r in rois:
                inp = preprocess_effnet(r)
                f = effnet(inp).cpu().numpy().reshape(-1)
                f_list.append(f)
                if effnet_cls is not None:
                    lg = effnet_cls(inp)
                    p = torch.softmax(lg, dim=1).cpu().numpy().reshape(-1)
                    p_list.append(p)
            feats = np.mean(np.stack(f_list, axis=0), axis=0)
            eff_proba = np.mean(np.stack(p_list, axis=0), axis=0) if len(p_list) > 0 else None
            try:
                logger.info(f"✓ EfficientNet feature length: {len(feats)}")
            except Exception:
                pass

        # Bangun blok fitur sesuai artefak training:
        #  - blok_1: EfficientNet 1280 + morfologi_mm (6) -> 1286, di-scale dengan scaler_combined
        #  - blok_2: morfologi_px (6) -> tidak di-scale
        #  - blok_3: probabilitas EfficientNet classifier (13) -> tidak di-scale
        morph_mm = np.array([
            panjang_mm, lebar_mm, keliling_mm, panjang_tulang_mm, morph["rasio_bentuk"], float(best.get("confidence", 0.0)),
        ], dtype=np.float32)
        # Hitung fitur px berdasar mm
        panjang_px = panjang_mm / MM_PER_PX
        lebar_px = lebar_mm / MM_PER_PX
        keliling_px = keliling_mm / MM_PER_PX
        panjang_tulang_px = panjang_tulang_mm / MM_PER_PX
        luas_px2 = morph.get("luas_mm2", 0.0) / (MM_PER_PX ** 2)
        morph_px = np.array([
            panjang_px, lebar_px, keliling_px, panjang_tulang_px, morph["rasio_bentuk"], float(best.get("confidence", 0.0)),
        ], dtype=np.float32)
        
        # Debug: Print feature dimensions
        logger.info(f"✓ EfficientNet features: {len(feats)}, Morphological mm: {len(morph_mm)}, px: {len(morph_px)}")
        # blok_1: scale
        block1 = np.concatenate([feats.astype(np.float32), morph_mm], axis=0)
        if scaler_combined is not None:
            try:
                expected = int(getattr(scaler_combined, 'n_features_in_', block1.shape[0]))
            except Exception:
                expected = block1.shape[0]
            cur = int(block1.shape[0])
            if cur != expected:
                if cur > expected:
                    logger.warning(f"⚠ block1 length {cur} > expected {expected}; truncating")
                    block1 = block1[:expected]
                else:
                    logger.warning(f"⚠ block1 length {cur} < expected {expected}; zero-padding")
                    pad = np.zeros((expected - cur,), dtype=np.float32)
                    block1 = np.concatenate([block1, pad], axis=0)
            try:
                if hasattr(scaler_combined, 'center_') and hasattr(scaler_combined, 'scale_'):
                    X_block1 = ((block1 - scaler_combined.center_) / scaler_combined.scale_).astype(np.float32)
                else:
                    X_block1 = scaler_combined.transform(block1.reshape(1, -1)).reshape(-1)
            except Exception as e:
                logger.warning(f"⚠ scaler_combined transform failed: {e}; using identity")
                X_block1 = block1.reshape(-1)
        else:
            X_block1 = block1.reshape(-1)
        # blok_3: probas dari EfficientNet classifier
        if eff_proba is None:
            eff_proba = np.zeros(len(CLASS_NAMES) or 13, dtype=np.float32)
        block2 = morph_px.astype(np.float32)
        block3 = eff_proba.astype(np.float32)
        full_combined = np.concatenate([X_block1, block2, block3], axis=0)
        logger.info(f"✓ Full combined length before selection: {len(full_combined)}")
        # Scale full vector for original pipeline if scaler is provided
        if scaler is not None:
            fc = full_combined.astype(np.float32)
            try:
                expected_all = int(getattr(scaler, 'n_features_in_', fc.shape[0]))
            except Exception:
                expected_all = fc.shape[0]
            cur_all = int(fc.shape[0])
            if cur_all != expected_all:
                if cur_all > expected_all:
                    logger.warning(f"⚠ full_combined length {cur_all} > scaler expected {expected_all}; truncating")
                    fc = fc[:expected_all]
                else:
                    logger.warning(f"⚠ full_combined length {cur_all} < scaler expected {expected_all}; zero-padding")
                    pad3 = np.zeros((expected_all - cur_all,), dtype=np.float32)
                    fc = np.concatenate([fc, pad3], axis=0)
            try:
                if hasattr(scaler, 'center_') and hasattr(scaler, 'scale_'):
                    full_scaled = ((fc - scaler.center_) / scaler.scale_).astype(np.float32)
                else:
                    full_scaled = scaler.transform(fc.reshape(1, -1)).reshape(-1)
            except Exception as e:
                logger.warning(f"⚠ scaler.transform(full) failed: {e}; using unscaled full_combined")
                full_scaled = fc.reshape(-1)
        else:
            full_scaled = full_combined.astype(np.float32)
        # Apply selector jika ada
        # Siapkan dua representasi:
        #  - X_combined: hanya blok_1 (1286) yang di-scale, untuk model combined
        #  - X_selected: selector.transform(full_combined) untuk model final yang memakai feature selection
        X_combined = X_block1.reshape(1, -1)
        if 'selector' in globals() and selector is not None:
            try:
                expected_sel = int(getattr(selector, 'n_features_in_', full_scaled.shape[0]))
            except Exception:
                expected_sel = full_scaled.shape[0]
            cur_sel = int(full_scaled.shape[0])
            if cur_sel != expected_sel:
                if cur_sel > expected_sel:
                    logger.warning(f"⚠ full_combined length {cur_sel} > selector expected {expected_sel}; truncating")
                    full_scaled = full_scaled[:expected_sel]
                else:
                    logger.warning(f"⚠ full_combined length {cur_sel} < selector expected {expected_sel}; zero-padding")
                    pad2 = np.zeros((expected_sel - cur_sel,), dtype=np.float32)
                    full_scaled = np.concatenate([full_scaled, pad2], axis=0)
            try:
                if hasattr(selector, 'get_support'):
                    mask = selector.get_support()
                    X_selected = full_scaled[mask].reshape(1, -1)
                    logger.info(f"✓ After selector length: {X_selected.shape[1]}")
                else:
                    X_selected = full_scaled.reshape(1, -1)
            except Exception as e:
                logger.warning(f"⚠ Selector apply failed: {e}; using full_combined")
                X_selected = full_scaled.reshape(1, -1)
        else:
            X_selected = full_scaled.reshape(1, -1)

        # Prediksi XGBoost
        logger.info(f"USE_COMBINED = {USE_COMBINED}")
        try:
            logger.info(f"✓ CLASS_NAMES ({len(CLASS_NAMES)}): {CLASS_NAMES}")
        except Exception:
            pass
        if USE_EFF_ONLY and eff_proba is not None:
            proba_meta = None
            proba = eff_proba.reshape(-1)
            decision_rule = "efficientnet_only"
        elif USE_COMBINED:
            logger.info("Using combined pipeline")
            mdl = xgb_combined
            if mdl is None:
                return jsonify({"success": False, "error": "xgboost_combined_not_loaded"}), 500
            # Combined dilatih pada block1 yang sudah di-scale
            X_in = X_block1.reshape(1, -1)
            try:
                expected_mdl = int(getattr(mdl, 'n_features_in_', X_in.shape[1]))
            except Exception:
                expected_mdl = X_in.shape[1]
            cur_mdl = int(X_in.shape[1])
            if cur_mdl != expected_mdl:
                if cur_mdl > expected_mdl:
                    logger.warning(f"⚠ combined input length {cur_mdl} > model expected {expected_mdl}; truncating")
                    X_in = X_in[:, :expected_mdl]
                else:
                    logger.warning(f"⚠ combined input length {cur_mdl} < model expected {expected_mdl}; zero-padding")
                    padm = np.zeros((X_in.shape[0], expected_mdl - cur_mdl), dtype=np.float32)
                    X_in = np.concatenate([X_in, padm], axis=1)
            if calibrator is not None and hasattr(calibrator, "predict_proba"):
                try:
                    proba = calibrator.predict_proba(X_in)[0]
                except Exception:
                    proba = mdl.predict_proba(X_in)[0]
            else:
                proba = mdl.predict_proba(X_in)[0]
            proba_meta = proba.copy()
            decision_rule = "xgboost_combined"
        else:
            logger.info("Using original pipeline")
            # Original/final pipeline: gunakan fitur hasil selector (serasi dengan training 'final')
            mdl = xgb_original
            if mdl is None:
                return jsonify({"success": False, "error": "xgboost_original_not_loaded"}), 500
            X_in = X_selected
            try:
                expected_mdl = int(getattr(mdl, 'n_features_in_', X_in.shape[1]))
            except Exception:
                expected_mdl = X_in.shape[1]
            cur_mdl = int(X_in.shape[1])
            if cur_mdl != expected_mdl:
                if cur_mdl > expected_mdl:
                    logger.warning(f"⚠ original input length {cur_mdl} > model expected {expected_mdl}; truncating")
                    X_in = X_in[:, :expected_mdl]
                else:
                    logger.warning(f"⚠ original input length {cur_mdl} < model expected {expected_mdl}; zero-padding")
                    padm = np.zeros((X_in.shape[0], expected_mdl - cur_mdl), dtype=np.float32)
                    X_in = np.concatenate([X_in, padm], axis=1)
            if calibrator is not None and hasattr(calibrator, "predict_proba"):
                try:
                    proba = calibrator.predict_proba(X_in)[0]
                except Exception:
                    proba = mdl.predict_proba(X_in)[0]
            else:
                proba = mdl.predict_proba(X_in)[0]
            proba_meta = proba.copy()
            decision_rule = "xgboost_original"
        final_proba = eff_proba if eff_proba is not None else proba
        try:
            if not USE_EFF_ONLY and eff_proba is not None and proba_meta is not None:
                if META_FIRST:
                    final_proba = proba_meta
                    if EFF_BLEND_EPS > 0.0:
                        final_proba = ((1.0 - EFF_BLEND_EPS) * final_proba) + (EFF_BLEND_EPS * eff_proba)
                    decision_rule = "meta_first"
                else:
                    if USE_GATING:
                        idxs = np.argsort(proba_meta)[::-1]
                        m = float(proba_meta[idxs[0]] - proba_meta[idxs[1]]) if len(idxs) > 1 else 1.0
                        c1 = float(np.max(proba_meta))
                        if (m >= GATE_MARGIN) and (c1 >= GATE_CONF):
                            final_proba = (BLEND_META_WEIGHT * proba_meta) + ((1.0 - BLEND_META_WEIGHT) * eff_proba)
                            decision_rule = decision_rule + "+blend"
                        else:
                            w = float(BLEND_META_WEIGHT * 0.5)
                            final_proba = (w * proba_meta) + ((1.0 - w) * eff_proba)
                            decision_rule = "gate_blend_low"
                    else:
                        final_proba = (BLEND_META_WEIGHT * proba_meta) + ((1.0 - BLEND_META_WEIGHT) * eff_proba)
                try:
                    if CLASS_NAMES and len(CLASS_NAMES) == len(final_proba):
                        idxs_f = np.argsort(final_proba)[::-1]
                        if len(idxs_f) > 1:
                            i1, i2 = int(idxs_f[0]), int(idxs_f[1])
                            n1, n2 = CLASS_NAMES[i1], CLASS_NAMES[i2]
                            pairs = {("Leaf_Tanjung","Lingga"), ("Lingga","Leaf_Tanjung"), ("Mia","Leaf_Tanjung"), ("Leaf_Tanjung","Mia"), ("Inata_agrihorti","Branang"), ("Branang","Inata_agrihorti"), ("Branang","Hot_beauty"), ("Hot_beauty","Branang"), ("Ciko","Leaf_Tanjung"), ("Leaf_Tanjung","Ciko")}
                            ei = int(np.argmax(eff_proba)) if eff_proba is not None else i1
                            en = CLASS_NAMES[ei]
                            if (n1, n2) in pairs and en == n2:
                                def _sc(stats):
                                    def s(v, st):
                                        if isinstance(st, dict):
                                            mn = float(st.get("min", v))
                                            mx = float(st.get("max", v))
                                            mu = float(st.get("mean", v))
                                            sd = float(st.get("std", 1e-6))
                                            if v < mn or v > mx:
                                                return 0.0
                                            z = abs(v - mu) / (sd if sd > 1e-6 else 1.0)
                                            return 1.0 / (1.0 + z)
                                        return 0.0
                                    ms = []
                                    ms.append(s(panjang_mm, stats.get("panjang_daun_mm")))
                                    ms.append(s(lebar_mm, stats.get("lebar_daun_mm")))
                                    ms.append(s(keliling_mm, stats.get("keliling_daun_mm")))
                                    ms.append(s(panjang_tulang_mm, stats.get("panjang_tulang_daun_mm")))
                                    ms.append(s(float(morph["rasio_bentuk"]), stats.get("rasio_bentuk_daun")))
                                    return float(np.mean([m for m in ms if isinstance(m, (int, float))])) if len(ms) > 0 else 0.0
                                s1 = _sc(get_variety_characteristics(n1))
                                s2 = _sc(get_variety_characteristics(n2))
                                ec = float(np.max(eff_proba)) if eff_proba is not None else float(np.max(final_proba))
                                if (s2 - s1) >= 0.20 and ec >= 0.65:
                                    final_proba = (0.30 * proba_meta) + (0.70 * eff_proba)
                                    decision_rule = "pair_tiebreak_eff_morph"
                except Exception:
                    pass
                try:
                    if CLASS_NAMES and len(CLASS_NAMES) == len(proba_meta):
                        idxs = np.argsort(proba_meta)[::-1]
                        if len(idxs) > 1:
                            i1, i2 = int(idxs[0]), int(idxs[1])
                            n1, n2 = CLASS_NAMES[i1], CLASS_NAMES[i2]
                            pairs = {("Leaf_Tanjung","Lingga"), ("Lingga","Leaf_Tanjung"), ("Mia","Leaf_Tanjung"), ("Leaf_Tanjung","Mia"), ("Inata_agrihorti","Branang"), ("Branang","Inata_agrihorti"), ("Branang","Hot_beauty"), ("Hot_beauty","Branang"), ("Ciko","Leaf_Tanjung"), ("Leaf_Tanjung","Ciko")}
                            c_top = float(np.max(proba_meta))
                            if (n1, n2) in pairs and c_top < 0.68:
                                def score_for(stats):
                                    def s(v, st):
                                        if isinstance(st, dict):
                                            mn = float(st.get("min", v))
                                            mx = float(st.get("max", v))
                                            mu = float(st.get("mean", v))
                                            sd = float(st.get("std", 1e-6))
                                            if v < mn or v > mx:
                                                return 0.0
                                            z = abs(v - mu) / (sd if sd > 1e-6 else 1.0)
                                            return 1.0 / (1.0 + z)
                                        return 0.0
                                    ms = []
                                    ms.append(s(panjang_mm, stats.get("panjang_daun_mm")))
                                    ms.append(s(lebar_mm, stats.get("lebar_daun_mm")))
                                    ms.append(s(keliling_mm, stats.get("keliling_daun_mm")))
                                    ms.append(s(panjang_tulang_mm, stats.get("panjang_tulang_daun_mm")))
                                    ms.append(s(float(morph["rasio_bentuk"]), stats.get("rasio_bentuk_daun")))
                                    return float(np.mean([m for m in ms if isinstance(m, (int, float))])) if len(ms) > 0 else 0.0
                                s1 = score_for(get_variety_characteristics(n1))
                                s2 = score_for(get_variety_characteristics(n2))
                                ei = int(np.argmax(eff_proba))
                                en = CLASS_NAMES[ei]
                                if (s2 - s1) >= 0.25 and en == n2 and xgb_combined is not None:
                                    X_in = X_block1.reshape(1, -1)
                                    try:
                                        expected_mdl = int(getattr(xgb_combined, 'n_features_in_', X_in.shape[1]))
                                    except Exception:
                                        expected_mdl = X_in.shape[1]
                                    cur_mdl = int(X_in.shape[1])
                                    if cur_mdl != expected_mdl:
                                        if cur_mdl > expected_mdl:
                                            X_in = X_in[:, :expected_mdl]
                                        else:
                                            padm = np.zeros((X_in.shape[0], expected_mdl - cur_mdl), dtype=np.float32)
                                            X_in = np.concatenate([X_in, padm], axis=1)
                                    if calibrator is not None and hasattr(calibrator, "predict_proba"):
                                        try:
                                            proba_c = calibrator.predict_proba(X_in)[0]
                                        except Exception:
                                            proba_c = xgb_combined.predict_proba(X_in)[0]
                                    else:
                                        proba_c = xgb_combined.predict_proba(X_in)[0]
                                    final_proba = (0.50 * proba_c) + (0.50 * eff_proba)
                                    decision_rule = "targeted_combined_tiebreak"
                            elif (n1, n2) in pairs and c_top < 0.90:
                                def score_for(stats):
                                    def s(v, st):
                                        if isinstance(st, dict):
                                            mn = float(st.get("min", v))
                                            mx = float(st.get("max", v))
                                            mu = float(st.get("mean", v))
                                            sd = float(st.get("std", 1e-6))
                                            if v < mn or v > mx:
                                                return 0.0
                                            z = abs(v - mu) / (sd if sd > 1e-6 else 1.0)
                                            return 1.0 / (1.0 + z)
                                        return 0.0
                                    ms = []
                                    ms.append(s(panjang_mm, stats.get("panjang_daun_mm")))
                                    ms.append(s(lebar_mm, stats.get("lebar_daun_mm")))
                                    ms.append(s(keliling_mm, stats.get("keliling_daun_mm")))
                                    ms.append(s(panjang_tulang_mm, stats.get("panjang_tulang_daun_mm")))
                                    ms.append(s(float(morph["rasio_bentuk"]), stats.get("rasio_bentuk_daun")))
                                    return float(np.mean([m for m in ms if isinstance(m, (int, float))])) if len(ms) > 0 else 0.0
                                s1 = score_for(get_variety_characteristics(n1))
                                s2 = score_for(get_variety_characteristics(n2))
                                ei = int(np.argmax(eff_proba))
                                en = CLASS_NAMES[ei]
                                if (s2 - s1) >= 0.30 and en == n2:
                                    final_proba = (0.50 * proba_meta) + (0.50 * eff_proba)
                                    decision_rule = "targeted_morph_eff_tiebreak"
                except Exception:
                    pass
                
                try:
                    if eff_proba is not None and proba_meta is not None and CLASS_NAMES and len(CLASS_NAMES) == len(eff_proba):
                        ei = int(np.argmax(eff_proba))
                        mi = int(np.argmax(proba_meta))
                        en = CLASS_NAMES[ei]
                        mn = CLASS_NAMES[mi]
                        def _sc(nm):
                            st = get_variety_characteristics(nm)
                            ms = []
                            def s(v, st):
                                if isinstance(st, dict):
                                    mnv = float(st.get("min", v))
                                    mxv = float(st.get("max", v))
                                    muv = float(st.get("mean", v))
                                    sdv = float(st.get("std", 1e-6))
                                    if v < mnv or v > mxv:
                                        return 0.0
                                    z = abs(v - muv) / (sdv if sdv > 1e-6 else 1.0)
                                    return 1.0 / (1.0 + z)
                                return 0.0
                            ms.append(s(panjang_mm, st.get("panjang_daun_mm")))
                            ms.append(s(lebar_mm, st.get("lebar_daun_mm")))
                            ms.append(s(keliling_mm, st.get("keliling_daun_mm")))
                            ms.append(s(panjang_tulang_mm, st.get("panjang_tulang_daun_mm")))
                            ms.append(s(float(morph["rasio_bentuk"]), st.get("rasio_bentuk_daun")))
                            return float(np.mean([m for m in ms if isinstance(m, (int, float))])) if len(ms) > 0 else 0.0
                        es = _sc(en)
                        ms2 = _sc(mn)
                        ec = float(np.max(eff_proba))
                        idxs_eff = np.argsort(eff_proba)[::-1]
                        em = float(eff_proba[idxs_eff[0]] - eff_proba[idxs_eff[1]]) if len(idxs_eff) > 1 else 1.0
                        focus = {"Ciko","Inata_agrihorti","Branang","Mia"}
                        if (en in focus) and (ei != mi):
                            if (es - ms2) >= 0.12 and ec >= 0.62 and em >= 0.08:
                                final_proba = (0.35 * proba_meta) + (0.65 * eff_proba)
                                decision_rule = "morph_tiebreak_targeted"
                except Exception:
                    pass
        except Exception:
            pass

        # Hitung prediksi, lalu lakukan boosting berbasis karakteristik varietas
        pred_idx = int(np.argmax(final_proba))
        try:
            def score_for(stats):
                def s(v, st):
                    if isinstance(st, dict):
                        mn = float(st.get("min", v))
                        mx = float(st.get("max", v))
                        mu = float(st.get("mean", v))
                        sd = float(st.get("std", 1e-6))
                        if v < mn or v > mx:
                            return 0.0
                        z = abs(v - mu) / (sd if sd > 1e-6 else 1.0)
                        return 1.0 / (1.0 + z)
                    return 0.0
                ms = []
                ms.append(s(panjang_mm, stats.get("panjang_daun_mm")))
                ms.append(s(lebar_mm, stats.get("lebar_daun_mm")))
                ms.append(s(keliling_mm, stats.get("keliling_daun_mm")))
                ms.append(s(panjang_tulang_mm, stats.get("panjang_tulang_daun_mm")))
                ms.append(s(float(morph["rasio_bentuk"]), stats.get("rasio_bentuk_daun")))
                return float(np.mean([m for m in ms if isinstance(m, (int, float))])) if len(ms) > 0 else 0.0

            if CLASS_NAMES and len(CLASS_NAMES) == len(final_proba):
                boosts = []
                for k, nm in enumerate(CLASS_NAMES):
                    stats = get_variety_characteristics(nm)
                    boosts.append(score_for(stats))
                boosts = np.array(boosts, dtype=np.float32)
                beta = MORPH_BETA
                adj = (1.0 + beta * boosts)
                if MORPH_SUPPRESS > 0.0:
                    adj = adj * (1.0 - MORPH_SUPPRESS * (1.0 - boosts))
                final_proba = final_proba * adj
                sm = np.sum(final_proba)
                if sm > 0:
                    final_proba = final_proba / sm
                try:
                    idxs = np.argsort(final_proba)[::-1]
                    if len(idxs) > 1:
                        i1, i2 = int(idxs[0]), int(idxs[1])
                        n1, n2 = CLASS_NAMES[i1], CLASS_NAMES[i2]
                        m2 = float(final_proba[i1] - final_proba[i2])
                        ei = int(np.argmax(eff_proba)) if eff_proba is not None else i1
                        en = CLASS_NAMES[ei]
                        focus = {"Mia","Inata_agrihorti","Ciko"}
                        if (n1 == "Leaf_Tanjung" and n2 in focus and en == n2 and m2 < 0.060):
                            s1 = score_for(get_variety_characteristics(n1))
                            s2 = score_for(get_variety_characteristics(n2))
                            if (s2 - s1) >= 0.15:
                                final_proba[i2] = final_proba[i2] * 1.08
                                sm2 = np.sum(final_proba)
                                if sm2 > 0:
                                    final_proba = final_proba / sm2
                        if (n1 == "Branang" and n2 == "Inata_agrihorti" and en == n2 and m2 < 0.065):
                            s1 = score_for(get_variety_characteristics(n1))
                            s2 = score_for(get_variety_characteristics(n2))
                            if (s2 - s1) >= 0.18:
                                final_proba[i2] = final_proba[i2] * 1.06
                                sm2 = np.sum(final_proba)
                                if sm2 > 0:
                                    final_proba = final_proba / sm2
                except Exception:
                    pass
                
            if isinstance(PROBA_ALPHA, float) and PROBA_ALPHA != 1.0:
                fp = np.power(np.clip(final_proba, 1e-8, 1.0), PROBA_ALPHA)
                s2 = np.sum(fp)
                if s2 > 0:
                    final_proba = fp / s2
        except Exception:
            pass
        
        pred_idx = int(np.argmax(final_proba))
        conf = float(np.max(final_proba))
        if CLASS_NAMES and 0 <= pred_idx < len(CLASS_NAMES):
            variety_name = CLASS_NAMES[pred_idx]
        else:
            variety_name = f"Varietas_{pred_idx}"
        characteristics = get_variety_characteristics(variety_name)

        return jsonify({
            "success": True,
            "api_version": API_VERSION,
            "decision_rule": decision_rule,
            "yolo": {"detections": dets},
            "variety": variety_name,
            "variety_code": pred_idx,
            "confidence": conf,
            "confidence_percentage": f"{conf*100:.2f}%",
            "morphology_info": {
                "panjang_daun_mm": float(round(panjang_mm, 2)),
                "lebar_daun_mm": float(round(lebar_mm, 2)),
                "keliling_daun_mm": float(round(keliling_mm, 2)),
                "panjang_tulang_daun_mm": float(round(panjang_tulang_mm, 2)),
                "rasio_bentuk_daun": float(morph["rasio_bentuk"]),
                "scale_mm_per_px": float(MM_PER_PX if MORPH_UNITS == "mm" else 1.0),
            },
            "measurement_quality": morph.get("measurement_quality", {"status": "ok", "issues": []}),
            "variety_characteristics": characteristics,
        }), 200
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@app.route("/", methods=["GET"]) 
def index():
    return jsonify({
        "message": "Variety classification API",
        "version": API_VERSION,
        "endpoints": ["GET /api/health", "POST /predict"],
    })


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
