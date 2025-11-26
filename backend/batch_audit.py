import os
import csv
import time
import shutil
import requests
from typing import List, Tuple
API_URL = os.environ.get("API_URL", "http://127.0.0.1:5000/predict")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.70"))
INPUT_ROOT = os.environ.get(
    "INPUT_ROOT",
    r"D:\DOCUMENT\Hasna punya\Semester 7\variety_classification\gambar test\ruang lingkup"
)
SKIP_COPY = str(os.environ.get("SKIP_COPY", "0")).lower() in ("1","true","yes","y")
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0") or "0")
GROUP_FOLDERS = str(os.environ.get("GROUP_FOLDERS", "0")).lower() in ("1","true","yes","y")
APPLY_GROUP = str(os.environ.get("APPLY_GROUP", "0")).lower() in ("1","true","yes","y")
RATIO_THRESHOLD = float(os.environ.get("RATIO_THRESHOLD", "0.70"))
FOLDER_CONF_MIN = float(os.environ.get("FOLDER_CONF_MIN", "0.70"))
REQUEST_FORCE_FULL = str(os.environ.get("REQUEST_FORCE_FULL", "")).lower() in ("1","true","yes","y")
REQUEST_USE_EXPANDED_ROI = str(os.environ.get("REQUEST_USE_EXPANDED_ROI", "")).lower() in ("1","true","yes","y")
REQUEST_IGNORE_YOLO = str(os.environ.get("REQUEST_IGNORE_YOLO", "")).lower() in ("1","true","yes","y")
REQUEST_ROI_MARGIN = os.environ.get("REQUEST_ROI_MARGIN", "")

DATASET_CLASSES = os.path.join(
    os.getcwd(),
    "data",
    "Dataset project.v6i.multiclass",
    "train",
    "_classes.csv",
)


def load_verified_class_names(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        return [
            "Branang",
            "Carla_agrihorti",
            "Carvi_agrihorti",
            "Ciko",
            "Hot_beauty",
            "Hot_vision",
            "Inata_agrihorti",
            "Ivegri",
            "Leaf_Tanjung",
            "Lingga",
            "Mia",
            "Pertiwi",
            "Pilar",
        ]
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header[1:]


def is_image_file(name: str) -> bool:
    e = name.lower()
    return e.endswith((".jpg", ".jpeg", ".png", ".bmp"))


def gather_images(root: str) -> List[str]:
    items = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if is_image_file(fn):
                items.append(os.path.join(dirpath, fn))
    if MAX_IMAGES and MAX_IMAGES > 0:
        items.sort()
        return items[:MAX_IMAGES]
    return items


def parse_confidence(conf_str) -> float:
    try:
        if isinstance(conf_str, str) and conf_str.endswith("%"):
            return float(conf_str.rstrip("%")) / 100.0
        return float(conf_str)
    except Exception:
        return 0.0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_to(target_dir: str, src_path: str):
    ensure_dir(target_dir)
    base = os.path.basename(src_path)
    dst = os.path.join(target_dir, base)
    if os.path.exists(dst):
        name, ext = os.path.splitext(base)
        dst = os.path.join(target_dir, f"{name}_{int(time.time())}{ext}")
    shutil.copy2(src_path, dst)
    return dst


def main():
    print("🚀 Batch audit dimulai…")
    print(f"API: {API_URL}")
    print(f"Input root: {INPUT_ROOT}")
    print(f"Threshold: {CONF_THRESHOLD}")

    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Path tidak ditemukan: {INPUT_ROOT}")
        return 1

    class_names = load_verified_class_names(DATASET_CLASSES)
    print(f"✓ Kelas terverifikasi ({len(class_names)}): {', '.join(class_names)}")

    images = gather_images(INPUT_ROOT)
    print(f"✓ Gambar ditemukan: {len(images)} file")
    if not images:
        print("⚠ Tidak ada file gambar untuk diproses")
        return 0

    out_root = INPUT_ROOT.rstrip("\\/") + "_grouped"
    ensure_dir(out_root)
    unsure_dir = os.path.join(out_root, "uncertain")
    if not SKIP_COPY:
        ensure_dir(unsure_dir)
        for cls in class_names:
            ensure_dir(os.path.join(out_root, cls))

    summary_csv = os.path.join(out_root, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_path", "predicted_variety", "confidence", "target_folder"]) 
        start = time.time()
        ok = 0
        fail = 0
        folder_stats = {}
        for i, img_path in enumerate(images, start=1):
            if i % 50 == 0:
                print(f"… {i}/{len(images)}")
            try:
                with open(img_path, "rb") as fp:
                    form = {}
                    form["force_full"] = "1" if REQUEST_FORCE_FULL else "0"
                    form["use_expanded_roi"] = "1" if REQUEST_USE_EXPANDED_ROI else "0"
                    if REQUEST_IGNORE_YOLO:
                        form["ignore_yolo"] = "1"
                    try:
                        rm = float(REQUEST_ROI_MARGIN)
                        if rm > 0:
                            form["roi_margin"] = str(rm)
                    except Exception:
                        pass
                    r = requests.post(API_URL, files={"image": fp}, data=form, timeout=90)
                if r.status_code != 200:
                    fail += 1
                    copy_to(unsure_dir, img_path)
                    w.writerow([img_path, "", 0.0, "uncertain"]) 
                    continue
                data = r.json()
                pred = str(data.get("variety", ""))
                conf = parse_confidence(data.get("confidence_percentage", 0.0))
                parent = os.path.dirname(img_path)
                d = folder_stats.setdefault(parent, {})
                if pred:
                    c = d.get(pred, {"count": 0, "conf_sum": 0.0})
                    c["count"] += 1
                    c["conf_sum"] += conf
                    d[pred] = c

                if (pred in class_names) and (conf >= CONF_THRESHOLD):
                    target = os.path.join(out_root, pred)
                    if not SKIP_COPY:
                        dst = copy_to(target, img_path)
                    w.writerow([img_path, pred, conf, target])
                    ok += 1
                else:
                    if not SKIP_COPY:
                        dst = copy_to(unsure_dir, img_path)
                    w.writerow([img_path, pred, conf, "uncertain"]) 
            except Exception:
                fail += 1
                if not SKIP_COPY:
                    copy_to(unsure_dir, img_path)
                w.writerow([img_path, "", 0.0, "uncertain"]) 
                continue

    dur = time.time() - start
    print(f"✅ Selesai. Keberhasilan: {ok}, gagal/uncertain: {fail}, waktu: {dur:.1f}s")
    print(f"📄 Ringkasan: {summary_csv}")
    print(f"📁 Output: {out_root}")

    folder_csv = os.path.join(out_root, "folder_summary.csv")
    with open(folder_csv, "w", newline="", encoding="utf-8") as fsum:
        w2 = csv.writer(fsum)
        w2.writerow(["folder_path", "total_images", "top1_variety", "top1_count", "top1_ratio", "top1_avg_conf", "top2_variety", "top2_count", "top2_ratio"]) 
        for folder, stats in folder_stats.items():
            total = sum(v["count"] for v in stats.values()) if stats else 0
            if total == 0:
                w2.writerow([folder, 0, "", 0, 0.0, 0.0, "", 0, 0.0])
                continue
            items = [(k, v["count"], v["conf_sum"]/max(1, v["count"])) for k, v in stats.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            top1 = items[0]
            top2 = items[1] if len(items) > 1 else ("", 0, 0.0)
            w2.writerow([
                folder,
                total,
                top1[0],
                top1[1],
                float(top1[1]) / float(total),
                top1[2],
                top2[0],
                top2[1],
                float(top2[1]) / float(total) if total > 0 else 0.0,
            ])
    print(f"📄 Ringkasan folder: {folder_csv}")

    rec_csv = os.path.join(out_root, "recommended_grouping.csv")
    with open(rec_csv, "w", newline="", encoding="utf-8") as frec:
        w3 = csv.writer(frec)
        w3.writerow(["folder_path", "suggested_variety", "top1_ratio", "top1_avg_conf", "decision", "reason"]) 
        for folder, stats in folder_stats.items():
            total = sum(v["count"] for v in stats.values()) if stats else 0
            if total == 0:
                w3.writerow([folder, "", 0.0, 0.0, "skip", "no_images"])
                continue
            items = [(k, v["count"], v["conf_sum"]/max(1, v["count"])) for k, v in stats.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            top1 = items[0]
            ratio = float(top1[1]) / float(total)
            avgc = float(top1[2])
            decision = "uncertain"
            reason = "below_threshold"
            if (top1[0] in class_names) and (ratio >= RATIO_THRESHOLD) and (avgc >= FOLDER_CONF_MIN):
                decision = "assign"
                reason = "majority_and_confidence_ok"
            w3.writerow([folder, top1[0], ratio, avgc, decision, reason])
    print(f"📄 Rekomendasi grouping: {rec_csv}")

    if GROUP_FOLDERS and APPLY_GROUP:
        print("⚙️  Menerapkan grouping folder berdasarkan rekomendasi…")
        with open(rec_csv, "r", encoding="utf-8") as fr:
            reader = csv.DictReader(fr)
            rows = list(reader)
        # Fallback: jika rekomendasi kosong, bangun dari folder_summary.csv
        if len(rows) == 0:
            try:
                with open(folder_csv, "r", encoding="utf-8") as fsum:
                    rdr = csv.DictReader(fsum)
                    rows = []
                    for r in rdr:
                        try:
                            folder = r.get("folder_path")
                            var = r.get("top1_variety")
                            ratio = float(r.get("top1_ratio") or 0.0)
                            avgc = float(r.get("top1_avg_conf") or 0.0)
                            dec = "assign" if (var in class_names and ratio >= RATIO_THRESHOLD and avgc >= FOLDER_CONF_MIN) else "uncertain"
                            rows.append({
                                "folder_path": folder,
                                "suggested_variety": var,
                                "top1_ratio": ratio,
                                "top1_avg_conf": avgc,
                                "decision": dec,
                            })
                        except Exception:
                            continue
                # Tulis ulang rec_csv dari fallback
                with open(rec_csv, "w", newline="", encoding="utf-8") as frec2:
                    w3 = csv.writer(frec2)
                    w3.writerow(["folder_path", "suggested_variety", "top1_ratio", "top1_avg_conf", "decision", "reason"]) 
                    for rr in rows:
                        w3.writerow([
                            rr.get("folder_path"), rr.get("suggested_variety"), rr.get("top1_ratio"), rr.get("top1_avg_conf"), rr.get("decision"),
                            "majority_and_confidence_ok" if rr.get("decision") == "assign" else "below_threshold"
                        ])
            except Exception:
                rows = []
        for row in rows:
            if str(row.get("decision")) != "assign":
                continue
            folder = str(row.get("folder_path"))
            var = str(row.get("suggested_variety"))
            if not folder or not var or var not in class_names:
                continue
            target_base = os.path.join(out_root, var, os.path.basename(folder))
            ensure_dir(target_base)
            try:
                for fn in os.listdir(folder):
                    src = os.path.join(folder, fn)
                    if os.path.isfile(src) and is_image_file(fn):
                        dst = os.path.join(target_base, fn)
                        if os.path.exists(dst):
                            name, ext = os.path.splitext(fn)
                            dst = os.path.join(target_base, f"{name}_{int(time.time())}{ext}")
                        shutil.copy2(src, dst) if SKIP_COPY else shutil.move(src, dst)
            except Exception:
                continue
        print("✅ Grouping folder diterapkan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
