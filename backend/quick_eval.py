import os
import csv
import random
import time
from statistics import mean
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:5000/predict")
DATASET = os.environ.get("DATASET", os.path.join(os.getcwd(), "data", "Dataset project.v6i.multiclass", "test"))
EVAL_ALL = str(os.environ.get("EVAL_ALL", "0")).lower() in ("1","true","yes","y")
PER_CLASS = int(os.environ.get("PER_CLASS", "4"))
CLASSES_CSV = os.path.join(DATASET, "_classes.csv")

with open(CLASSES_CSV, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    class_names = header[1:]
    rows = list(reader)

items = []
for row in rows:
    filename = row[0]
    vec = [int(x) for x in row[1:]]
    true_idx = vec.index(1)
    items.append((os.path.join(DATASET, filename), true_idx))

random.seed(42)
by_class = {}
for p, idx in items:
    by_class.setdefault(idx, []).append(p)

if EVAL_ALL:
    sample = [(p, idx) for idx, paths in by_class.items() for p in paths]
else:
    sample = []
    for idx, paths in by_class.items():
        random.shuffle(paths)
        sample.extend([(p, idx) for p in paths[:PER_CLASS]])
    sample = sample[:50]

ok = 0
conf = []
miss = []
start = time.time()
for path, true_idx in sample:
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        r = requests.post(API_URL, files={"image": f})
    if r.status_code != 200:
        miss.append((os.path.basename(path), "http_error"))
        continue
    js = r.json()
    pred_name = js.get("variety")
    conf_pct = js.get("confidence_percentage", "0%")
    try:
        cp = float(str(conf_pct).strip().rstrip("%"))
    except Exception:
        cp = 0.0
    conf.append(cp)
    if pred_name in class_names:
        pred_idx = class_names.index(pred_name)
        ok += int(pred_idx == true_idx)
    else:
        miss.append((os.path.basename(path), pred_name))

acc = ok / max(1, len(sample))
label = "FullEval" if EVAL_ALL else "QuickEval"
print(f"{label}: {ok}/{len(sample)} correct, accuracy={acc*100:.2f}% | avg_conf={mean(conf) if conf else 0:.2f}%")
if miss:
    print("Mislabels (first 10):", miss[:10])
print(f"Elapsed: {time.time()-start:.1f}s")
