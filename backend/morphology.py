# =============================================================================
# BAGIAN 0: INSTALL SEMUA MODUL YANG DIPERLUKAN
# =============================================================================
print("🔄 Menginstal modul yang diperlukan...")

# Install roboflow
try:
    import roboflow
    print("✅ Modul roboflow sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul roboflow...")
    !pip install roboflow -q
    import roboflow
    print("✅ Modul roboflow berhasil diinstal")

# Install ultralytics
try:
    from ultralytics import YOLO
    print("✅ Modul ultralytics sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul ultralytics...")
    !pip install ultralytics -q
    from ultralytics import YOLO
    print("✅ Modul ultralytics berhasil diinstal")

# Install opencv-contrib-python (untuk ximgproc)
try:
    import cv2.ximgproc
    print("✅ Modul cv2.ximgproc sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul opencv-contrib-python...")
    !pip install opencv-contrib-python -q
    import cv2.ximgproc
    print("✅ Modul opencv-contrib-python berhasil diinstal")

# Install scikit-image untuk skeletonization yang lebih baik
try:
    from skimage.morphology import skeletonize
    print("✅ Modul scikit-image sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul scikit-image...")
    !pip install scikit-image -q
    from skimage.morphology import skeletonize
    print("✅ Modul scikit-image berhasil diinstal")

# Install seaborn untuk visualisasi yang lebih baik
try:
    import seaborn as sns
    print("✅ Modul seaborn sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul seaborn...")
    !pip install seaborn -q
    import seaborn as sns
    print("✅ Modul seaborn berhasil diinstal")

# Install tqdm
try:
    from tqdm import tqdm
    print("✅ Modul tqdm sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul tqdm...")
    !pip install tqdm -q
    from tqdm import tqdm
    print("✅ Modul tqdm berhasil diinstal")

# Install PyYAML
try:
    import yaml
    print("✅ Modul PyYAML sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul PyYAML...")
    !pip install PyYAML -q
    import yaml
    print("✅ Modul PyYAML berhasil diinstal")

# Install matplotlib untuk visualisasi
try:
    import matplotlib.pyplot as plt
    print("✅ Modul matplotlib sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul matplotlib...")
    !pip install matplotlib -q
    import matplotlib.pyplot as plt
    print("✅ Modul matplotlib berhasil diinstal")

# Install scipy untuk analisis statistik
try:
    from scipy.stats import f_oneway
    print("✅ Modul scipy sudah terinstal")
except ImportError:
    print("🔄 Menginstal modul scipy...")
    !pip install scipy -q
    from scipy.stats import f_oneway
    print("✅ Modul scipy berhasil diinstal")

print("✅ Semua modul berhasil diinstal dan diimpor")

# =============================================================================
# BAGIAN 0.5: IMPORT SEMUA MODUL YANG DIPERLUKAN
# =============================================================================
import os
import cv2
import numpy as np
import pandas as pd
import random
import shutil
from datetime import datetime
import glob
from roboflow import Roboflow
from tqdm import tqdm
import json
import yaml
from google.colab import files
from skimage.morphology import skeletonize
import seaborn as sns
from collections import deque

print("✅ Semua modul berhasil diimpor")

# =============================================================================
# BAGIAN 1: UPLOAD FILE BEST.PT
# =============================================================================
def upload_best_pt():
    """Upload file best.pt dari komputer lokal"""
    print("📤 Silakan upload file model YOLO (best.pt):")
    uploaded = files.upload()
    if not uploaded:
        print("❌ Tidak ada file best.pt yang diupload!")
        return None
    filename = list(uploaded.keys())[0]
    if filename.endswith('.pt'):
        if filename != 'best.pt':
            print(f"🔄 Mengubah nama file dari '{filename}' menjadi 'best.pt'")
            os.rename(filename, 'best.pt')
        print("✅ File model YOLO berhasil diupload!")
        return 'best.pt'
    else:
        print(f"❌ File yang diupload ({filename}) bukan file model YOLO (.pt)!")
        return None

# =============================================================================
# BAGIAN 2: UPLOAD FILE VALIDASI (OPSIONAL)
# =============================================================================
def upload_validation_file():
    """Upload file validasi results.csv dari komputer lokal (opsional)"""
    print("📤 Silakan upload file validasi (results.csv, opsional):")
    uploaded = files.upload()
    if not uploaded:
        print("⚠ Tidak ada file validasi yang diupload, melanjutkan tanpa file validasi.")
        return None
    filename = list(uploaded.keys())[0]
    if filename.endswith('.csv'):
        if filename != 'results.csv':
            print(f"🔄 Mengubah nama file dari '{filename}' menjadi 'results.csv'")
            os.rename(filename, 'results.csv')
        print("✅ File validasi berhasil diupload!")
        return 'results.csv'
    else:
        print(f"❌ File yang diupload ({filename}) bukan file CSV!")
        return None

# =============================================================================
# BAGIAN 3: FUNGSI UNTUK MENGAMBIL DATASET DARI ROBOFLOW
# =============================================================================
def download_roboflow_dataset():
    """Download dataset dari Roboflow dengan parameter yang sudah ditentukan"""
    print("🔄 Mengunduh dataset dari Roboflow...")
    try:
        rf = Roboflow(api_key="nIo1pdhX5jI86T4ouK9y")
        project = rf.workspace("chili-canopy").project("hasna-project-0pgjl")
        version = project.version(6)
        dataset = version.download("yolov8")
        print(f"✅ Dataset berhasil diunduh dari: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"❌ Error mengunduh dataset: {e}")
        return None

# =============================================================================
# BAGIAN 4: FUNGSI UNTUK MENGANALISIS STRUKTUR DATASET
# =============================================================================
def analyze_dataset_structure(dataset_path):
    """Menganalisis struktur dataset dan mendapatkan daftar kelas"""
    print("🔍 Menganalisis struktur dataset...")
    valid_path = os.path.join(dataset_path, "valid")
    if not os.path.exists(valid_path):
        valid_path = os.path.join(dataset_path, "test")
    if not os.path.exists(valid_path):
        print("❌ Tidak ditemukan folder valid atau test")
        return None, None
    images_path = os.path.join(valid_path, "images")
    labels_path = os.path.join(valid_path, "labels")
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print("❌ Tidak ditemukan folder images atau labels")
        return None, None
    image_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                  glob.glob(os.path.join(images_path, "*.jpeg")) + \
                  glob.glob(os.path.join(images_path, "*.png"))
    label_files = glob.glob(os.path.join(labels_path, "*.txt"))
    print(f"✅ Ditemukan {len(image_files)} gambar dan {len(label_files)} label")
    yaml_path = os.path.join(dataset_path, "data.yaml")
    class_names = []
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            if 'names' in yaml_content:
                names_data = yaml_content['names']
                if isinstance(names_data, list):
                    class_names = names_data
                elif isinstance(names_data, dict):
                    class_names = [names_data[i] for i in range(len(names_data))]
        except Exception as e:
            print(f"⚠ Error parsing YAML: {e}")
    if not class_names:
        print("🔄 Mengekstrak nama kelas dari file label...")
        classes = set()
        image_to_class = {}
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    classes.add(class_id)
                    label_name = os.path.basename(label_file).replace('.txt', '')
                    image_to_class[label_name] = class_id
        class_names = [f"Class_{i}" for i in sorted(classes)]
    else:
        image_to_class = {}
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    label_name = os.path.basename(label_file).replace('.txt', '')
                    image_to_class[label_name] = class_id
    print(f"✅ Ditemukan {len(class_names)} kelas: {class_names}")
    return {
        'images_path': images_path,
        'labels_path': labels_path,
        'image_files': image_files,
        'label_files': label_files,
        'class_names': class_names,
        'image_to_class': image_to_class
    }, valid_path

# =============================================================================
# BAGIAN 5: FUNGSI UNTUK MEMILIH GAMBAR UNTUK PENGUJIAN
# =============================================================================
def select_test_images(dataset_info, images_per_class=16):
    """Memilih gambar untuk pengujian dengan jumlah tertentu per kelas"""
    print(f"🔍 Memilih {images_per_class} gambar per kelas untuk pengujian...")
    images_path = dataset_info['images_path']
    image_to_class = dataset_info['image_to_class']
    class_names = dataset_info['class_names']
    class_to_images = {}
    for img_name, class_id in image_to_class.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        if class_name not in class_to_images:
            class_to_images[class_name] = []
        for ext in ['jpg', 'jpeg', 'png']:
            img_path = os.path.join(images_path, f"{img_name}.{ext}")
            if os.path.exists(img_path):
                class_to_images[class_name].append((img_path, class_id))
                break
    selected_images = []
    for class_name in class_names:
        available_images = class_to_images.get(class_name, [])
        num_to_select = min(images_per_class, len(available_images))
        selected = random.sample(available_images, num_to_select) if available_images else []
        for img_path, class_id in selected:
            print(f"   ✅ Dipilih: {os.path.basename(img_path)} (Kelas: {class_name})")
        selected_images.extend([(img_path, class_id, class_name) for img_path, class_id in selected])
    print(f"✅ Total {len(selected_images)} gambar dipilih untuk pengujian")
    return selected_images

# =============================================================================
# BAGIAN 6: FUNGSI UNTUK MENYALIN GAMBAR UNTUK PENGUJIAN
# =============================================================================
def copy_test_images(selected_images, output_dir):
    """Menyalin gambar yang dipilih ke folder pengujian"""
    print(f"🔄 Menyalin gambar ke folder: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    copied_images = []
    for img_path, class_id, class_name in selected_images:
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)
        shutil.copy2(img_path, output_path)
        copied_images.append((output_path, class_id, class_name))
        print(f"   ✅ Disalin: {img_name} (Kelas: {class_name})")
    print(f"✅ {len(copied_images)} gambar berhasil disalin")
    return copied_images

# =============================================================================
# BAGIAN 7: FUNGSI UNTUK VALIDASI MODEL
# =============================================================================
def load_validation_results(validation_file):
    """Memuat dan menampilkan hasil validasi model dari file CSV"""
    print(f"🔍 Memuat hasil validasi dari: {validation_file}")
    try:
        val_data = pd.read_csv(validation_file)
        print("\n📊 Kolom yang tersedia dalam file validasi:")
        print(val_data.columns.tolist())

        possible_metrics = []
        for col in val_data.columns:
            if 'precision' in col.lower():
                possible_metrics.append(col)
            elif 'recall' in col.lower():
                possible_metrics.append(col)
            elif 'map' in col.lower():
                possible_metrics.append(col)

        if possible_metrics:
            print("\n📊 Metrik yang ditemukan:")
            print(val_data[['epoch'] + possible_metrics])
        else:
            print("\n⚠ Tidak ditemukan metrik validasi yang sesuai")
            print(val_data.head())

        return val_data
    except Exception as e:
        print(f"❌ Error memuat file validasi: {e}")
        return None

# =============================================================================
# BAGIAN 8: FUNGSI UNTUK ANALISIS MORFOLOGI (FINAL VERSION)
# =============================================================================

# =============================================================================
# FUNGSI HELPER BARU UNTUK MENGHITUNG PANJANG TULANG DAUN
# =============================================================================
def find_longest_skeleton_path(skeleton_img):
    """
    Menemukan jalur terpanjang pada gambar skeleton menggunakan BFS.
    Ini adalah pendekatan yang lebih akurat untuk mengukur panjang tulang daun utama.
    """
    points = np.argwhere(skeleton_img > 0)
    if len(points) < 2:
        return None, 0

    def bfs_farthest(start_point):
        queue = deque([(start_point[0], start_point[1], 0)])
        visited = {(start_point[0], start_point[1])}
        farthest_node = start_point
        max_dist = 0

        while queue:
            r, c, d = queue.popleft()
            if d > max_dist:
                max_dist = d
                farthest_node = (r, c)

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < skeleton_img.shape[0] and
                        0 <= nc < skeleton_img.shape[1] and
                        skeleton_img[nr, nc] > 0 and
                        (nr, nc) not in visited):
                        visited.add((nr, nc))
                        # PERBAIKAN: Mengubah cara menambahkan ke queue untuk menghindari error unpacking
                        queue.append((nr, nc, d + 1))
        return farthest_node, max_dist

    start_point = points[0]
    endpoint_A, _ = bfs_farthest(start_point)
    endpoint_B, path_length = bfs_farthest(endpoint_A)

    return (endpoint_A, endpoint_B), path_length

def load_yolo_model(best_pt_path):
    """Load model YOLOv8 custom untuk deteksi daun"""
    print("🔄 Memuat model YOLO...")
    try:
        if os.path.exists(best_pt_path):
            model = YOLO(best_pt_path)
            print(f"✅ Model custom berhasil dimuat dari: {best_pt_path}")
            return model
        else:
            print("❌ File best.pt tidak ditemukan")
            return None
    except Exception as e:
        print(f"❌ Error memuat model YOLO: {e}")
        return None

def detect_leaf_with_yolo(model, image_path, conf_threshold=0.25):
    """Deteksi daun menggunakan YOLO custom"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Gambar tidak ditemukan: {image_path}")
            return None, None, None, 0.0
        results = model(img, conf=conf_threshold)
        detections = results[0].boxes.data.cpu().numpy()
        if len(detections) == 0:
            print(f"   ❌ Tidak ada deteksi dengan conf={conf_threshold}")
            return None, None, None, 0.0
        detections_sorted = sorted(detections, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
        best_detection = detections_sorted[0]
        x1, y1, x2, y2, conf, cls = best_detection
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            print(f"   ⚠ Bounding box terlalu kecil untuk {os.path.basename(image_path)}")
            return None, None, None, 0.0
        leaf_crop = img[int(y1):int(y2), int(x1):int(x2)]
        result_img = img.copy()
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        class_name = model.names[int(cls)] if int(cls) in model.names else f"Class_{int(cls)}"
        print(f"   ✅ {class_name} terdeteksi: {int(x2-x1)}x{int(y2-y1)} px (Conf: {conf:.2f})")
        return result_img, leaf_crop, class_name, conf
    except Exception as e:
        print(f"   ❌ Error deteksi YOLO: {e}")
        return None, None, None, 0.0

# =============================================================================
# FUNGSI ANALISIS MORFOLOGI YANG SUDAH DIPERBAIKI (FINAL)
# =============================================================================
def analyze_leaf_morphology(image_path, yolo_model=None, scale_factor=0.1):
    """Analisis morfologi daun dengan perhitungan yang lebih robust."""
    print(f"🔍 Menganalisis: {os.path.basename(image_path)}")

    if yolo_model:
        result_img, leaf_crop, detected_variety, conf = detect_leaf_with_yolo(yolo_model, image_path)
        if leaf_crop is None:
            print(f"   ❌ Gagal mendeteksi daun di {os.path.basename(image_path)}")
            return None
    else:
        leaf_crop = cv2.imread(image_path)
        detected_variety = "Unknown"
        conf = 0.0

    try:
        gray = cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"   ⚠ Tidak ditemukan kontur untuk {os.path.basename(image_path)}")
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        panjang = max(w, h)
        lebar = min(w, h)

        if lebar < 10:
            print(f"   ⚠ Lebar terlalu kecil ({lebar}px), melewati gambar ini.")
            return None

        panjang_mm = round(panjang * scale_factor, 2)
        lebar_mm = round(lebar * scale_factor, 2)

        # --- PERBAIKAN 1: Menghitung Keliling dengan Batasan Wajar ---
        keliling_px = cv2.arcLength(largest_contour, True)
        keliling_mm = round(keliling_px * scale_factor, 2)

        # Batas maksimum keliling yang wajar (misal 2.2x keliling persegi panjang)
        max_keliling_wajar = 2.2 * (2 * (panjang_mm + lebar_mm))
        if keliling_mm > max_keliling_wajar:
            print(f"   ⚠ Keliling tidak wajar ({keliling_mm:.2f}mm), dibatasi menjadi {max_keliling_wajar:.2f}mm")
            keliling_mm = round(max_keliling_wajar, 2)

        # --- PERBAIKAN 2: Menghitung Panjang Tulang Daun dengan Fallback yang Kuat ---
        try:
            skeleton = skeletonize(thresh/255).astype(np.uint8) * 255
            path_info, panjang_tulang_px = find_longest_skeleton_path(skeleton)
            panjang_tulang_daun_mm = round(panjang_tulang_px * scale_factor, 2)

            # Jika hasil skeleton terlalu kecil (tidak masuk akal), gunakan fallback
            if panjang_tulang_daun_mm < (panjang_mm * 0.3):
                print(f"   ⚠ Panjang tulang tidak wajar ({panjang_tulang_daun_mm:.2f}mm), menggunakan estimasi")
                panjang_tulang_daun_mm = round(panjang_mm * 0.7, 2)

        except Exception as e:
            print(f"   ⚠ Error perhitungan tulang daun: {e}. Menggunakan estimasi.")
            panjang_tulang_daun_mm = round(panjang_mm * 0.7, 2)

        # --- PERBAIKAN 3: Menghitung Rasio Bentuk Daun ---
        rasio_bentuk_daun = round(lebar_mm / panjang_mm, 2) if panjang_mm > 0 else 0

        return {
            'gambar': os.path.basename(image_path),
            'varietas': detected_variety,
            'panjang_daun_mm': panjang_mm,
            'lebar_daun_mm': lebar_mm,
            'keliling_daun_mm': keliling_mm,
            'panjang_tulang_daun_mm': panjang_tulang_daun_mm,
            'rasio_bentuk_daun': rasio_bentuk_daun,
            'confidence': round(conf, 2)
        }
    except Exception as e:
        print(f"   ❌ Error ekstraksi fitur: {e}")
        return None

# =============================================================================
# BAGIAN 9: PROGRAM UTAMA
# =============================================================================
def main():
    print("🌿 DOWNLOAD, PILIH GAMBAR, DAN ANALISIS MORFOLOGI")
    print("="*60)

    OUTPUT_DIR = "/content/test_images"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_pt_path = upload_best_pt()
    if best_pt_path is None:
        print("❌ Tidak dapat melanjutkan tanpa file best.pt!")
        return

    validation_file = upload_validation_file()

    print("\n📋 STEP 0: Memuat hasil validasi model")
    if validation_file:
        val_data = load_validation_results(validation_file)
    else:
        print("⚠ Tidak ada file validasi diupload, melanjutkan tanpa file validasi")

    print("\n📋 STEP 1: Download dataset dari Roboflow")
    dataset_path = download_roboflow_dataset()
    if dataset_path is None:
        print("❌ Gagal mengunduh dataset")
        return

    print("\n📋 STEP 2: Analisis struktur dataset")
    dataset_info, valid_path = analyze_dataset_structure(dataset_path)
    if dataset_info is None:
        print("❌ Gagal menganalisis struktur dataset")
        return

    print("\n📋 STEP 3: Pilih gambar untuk pengujian")
    selected_images = select_test_images(dataset_info, images_per_class=16)

    print("\n📋 STEP 4: Salin gambar untuk pengujian")
    copied_images = copy_test_images(selected_images, OUTPUT_DIR)

    print("\n📋 STEP 5: Buat ringkasan")
    summary = {
        'timestamp': timestamp,
        'total_images': len(copied_images),
        'dataset_path': dataset_path,
        'valid_path': valid_path,
        'output_dir': OUTPUT_DIR,
        'images': []
    }
    for img_path, class_id, class_name in copied_images:
        summary['images'].append({
            'filename': os.path.basename(img_path),
            'path': img_path,
            'class_id': class_id,
            'class_name': class_name
        })
    summary_file = f"test_images_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Ringkasan disimpan ke: {summary_file}")
    print("\n📊 Ringkasan Gambar Uji:")
    print("-" * 60)
    print(f"{'No.':<5} {'Filename':<25} {'Kelas':<20} {'ID':<5}")
    print("-" * 60)
    for i, (img_path, class_id, class_name) in enumerate(copied_images, 1):
        print(f"{i:<5} {os.path.basename(img_path):<25} {class_name:<20} {class_id:<5}")
    print("-" * 60)

    print("\n📋 STEP 6: Analisis morfologi dengan YOLO")
    yolo_model = load_yolo_model(best_pt_path)
    if yolo_model:
        all_results = []
        for img_path, _, _ in copied_images:
            print(f"\n🔍 Memproses: {os.path.basename(img_path)}")
            result = analyze_leaf_morphology(img_path, yolo_model, scale_factor=0.1)
            if result:
                all_results.append(result)
                print(f"✅ Berhasil menganalisis: {os.path.basename(img_path)}")
            else:
                print(f"❌ Gagal menganalisis: {os.path.basename(img_path)}")
        if all_results:
            print("\n📊 Tabel Hasil Analisis Morfologi:")
            df = pd.DataFrame(all_results)
            # --- PERBAIKAN: Memperbarui urutan dan nama kolom ---
            kolom_urut = [
                'gambar', 'varietas', 'panjang_daun_mm', 'lebar_daun_mm',
                'keliling_daun_mm', 'panjang_tulang_daun_mm', 'rasio_bentuk_daun', 'confidence'
            ]
            df = df[kolom_urut]
            df.columns = [
                'Gambar', 'Varietas', 'Panjang Daun (mm)', 'Lebar Daun (mm)',
                'Keliling Daun (mm)', 'Panjang Tulang Daun (mm)', 'Rasio Bentuk Daun', 'Confidence'
            ]
            print(df.to_string(index=False))
            csv_filename = f'hasil_morfologi_{timestamp}.csv'
            df.to_csv(csv_filename, index=False)
            print(f"\n✅ Hasil disimpan ke '{csv_filename}'")
            print("\n📊 Interpretasi Fitur Morfologi:")
            print("-" * 60)
            print("1. Panjang daun: Ukuran dasar untuk identifikasi varietas")
            print("2. Lebar daun: Ukuran dasar untuk identifikasi varietas")
            print("3. Keliling daun: Menggambarkan bentuk tepi daun")
            print("4. Panjang tulang daun: Karakteristik struktur penting")
            # --- PERBAIKAN: Memperbarui teks interpretasi ---
            print("5. Rasio bentuk daun: Proporsi lebar terhadap panjang daun")
            print("6. Confidence: Keandalan deteksi varietas oleh model YOLO")
            print("-" * 60)
            print("\n🔍 Analisis Bentuk Daun:")
            for _, row in df.iterrows():
                # --- PERBAIKAN: Menggunakan rasio bentuk yang baru untuk klasifikasi ---
                rasio_bentuk = row['Rasio Bentuk Daun']
                if rasio_bentuk > 0.8:
                    bentuk = "Membulat"
                elif rasio_bentuk > 0.7:
                    bentuk = "Oval"
                else:
                    bentuk = "Lancip"
                print(f"- {row['Gambar']}: {bentuk} (Rasio: {rasio_bentuk:.2f}, Confidence: {row['Confidence']})")
            print("\n📊 Analisis Statistik (ANOVA) untuk Fitur Morfologi:")
            # --- PERBAIKAN: Memperbarui daftar fitur untuk ANOVA ---
            for fitur in ['Panjang Daun (mm)', 'Lebar Daun (mm)', 'Keliling Daun (mm)', 'Panjang Tulang Daun (mm)', 'Rasio Bentuk Daun']:
                groups = [group[fitur].values for name, group in df.groupby('Varietas')]
                stat, p_value = f_oneway(*groups)
                print(f"{fitur}: F-statistic={stat:.2f}, p-value={p_value:.4f}")
                if p_value < 0.05:
                    print(f"   ✅ Perbedaan signifikan untuk {fitur}")
                else:
                    print(f"   ⚠ Tidak ada perbedaan signifikan untuk {fitur}")

            print("\n📊 Membuat grafik perbandingan antar varietas...")
            varietas_grouped = df.groupby('Varietas')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Perbandingan Fitur Morfologi Antar Varietas', fontsize=16)
            # --- PERBAIKAN: Memperbarui daftar fitur untuk visualisasi ---
            fitur_list = [
                'Panjang Daun (mm)', 'Lebar Daun (mm)', 'Keliling Daun (mm)',
                'Panjang Tulang Daun (mm)', 'Rasio Bentuk Daun', 'Confidence'
            ]
            colors = plt.cm.tab10(np.linspace(0, 1, len(df['Varietas'].unique())))
            for i, fitur in enumerate(fitur_list):
                row = i // 3
                col = i % 3
                ax = axes[row, col]
                varietas_values = []
                varietas_names = []
                for varietas, group in varietas_grouped:
                    varietas_names.append(varietas)
                    varietas_values.append(group[fitur].values)
                bp = ax.boxplot(varietas_values, patch_artist=True)
                for j, (box, color) in enumerate(zip(bp['boxes'], colors)):
                    box.set_facecolor(color)
                ax.set_title(fitur)
                ax.set_xlabel('Varietas')
                ax.set_ylabel('Nilai')
                ax.set_xticklabels(varietas_names, rotation=45, ha='right')
                ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            grafik_filename = f'grafik_perbandingan_varietas_{timestamp}.png'
            plt.savefig(grafik_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Grafik disimpan ke: {grafik_filename}")

            fig, ax = plt.subplots(figsize=(12, 8))
            mean_df = df.groupby('Varietas')[fitur_list].mean().transpose()
            mean_df.plot(kind='bar', ax=ax, color=colors)
            ax.set_title('Rata-rata Fitur Morfologi per Varietas', fontsize=14)
            ax.set_xlabel('Fitur Morfologi')
            ax.set_ylabel('Rata-rata Nilai')
            ax.set_xticklabels(fitur_list, rotation=45, ha='right')
            ax.legend(title='Varietas')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            grafik_mean_filename = f'grafik_rata_rata_varietas_{timestamp}.png'
            plt.savefig(grafik_mean_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Grafik rata-rata disimpan ke: {grafik_mean_filename}")

            plt.figure(figsize=(10, 6))
            for varietas in df['Varietas'].unique():
                subset = df[df['Varietas'] == varietas]
                plt.scatter(subset['Panjang Daun (mm)'], subset['Lebar Daun (mm)'],
                           label=varietas, s=subset['Confidence']*100, alpha=0.6)
            plt.xlabel('Panjang Daun (mm)')
            plt.ylabel('Lebar Daun (mm)')
            plt.title('Panjang vs Lebar Daun per Varietas')
            plt.legend()
            plt.grid(True)
            scatter_filename = f'scatter_panjang_lebar_{timestamp}.png'
            plt.savefig(scatter_filename, dpi=300)
            plt.show()
            print(f"✅ Scatter plot disimpan ke: {scatter_filename}")

            print("\n📊 Membuat grafik nilai confidence antar varietas...")
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='Varietas', y='Confidence', palette='Set3')
            plt.title('Distribusi Nilai Confidence per Varietas', fontsize=14, fontweight='bold')
            plt.xlabel('Varietas', fontsize=12)
            plt.ylabel('Confidence', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            confidence_boxplot_filename = f'grafik_confidence_boxplot_{timestamp}.png'
            plt.savefig(confidence_boxplot_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Grafik boxplot confidence disimpan ke: {confidence_boxplot_filename}")

            plt.figure(figsize=(12, 6))
            mean_confidence = df.groupby('Varietas')['Confidence'].mean().sort_values(ascending=False)
            colors_conf = plt.cm.viridis(np.linspace(0, 1, len(mean_confidence)))
            bars = plt.bar(mean_confidence.index, mean_confidence.values, color=colors_conf, alpha=0.8)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}',
                         ha='center', va='bottom', fontweight='bold')
            plt.title('Rata-rata Nilai Confidence per Varietas', fontsize=14, fontweight='bold')
            plt.xlabel('Varietas', fontsize=12)
            plt.ylabel('Rata-rata Confidence', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1.1)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.tight_layout()
            confidence_bar_filename = f'grafik_confidence_rata_rata_{timestamp}.png'
            plt.savefig(confidence_bar_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Grafik rata-rata confidence disimpan ke: {confidence_bar_filename}")

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Hubungan Confidence dengan Fitur Morfologi', fontsize=16, fontweight='bold')
            axes[0, 0].scatter(df['Panjang Daun (mm)'], df['Confidence'], c=df['Confidence'], cmap='viridis', alpha=0.7, s=50)
            axes[0, 0].set_xlabel('Panjang Daun (mm)')
            axes[0, 0].set_ylabel('Confidence')
            axes[0, 0].set_title('Confidence vs Panjang Daun')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 1].scatter(df['Lebar Daun (mm)'], df['Confidence'], c=df['Confidence'], cmap='viridis', alpha=0.7, s=50)
            axes[0, 1].set_xlabel('Lebar Daun (mm)')
            axes[0, 1].set_ylabel('Confidence')
            axes[0, 1].set_title('Confidence vs Lebar Daun')
            axes[0, 1].grid(True, alpha=0.3)
            axes[1, 0].scatter(df['Panjang Tulang Daun (mm)'], df['Confidence'], c=df['Confidence'], cmap='viridis', alpha=0.7, s=50)
            axes[1, 0].set_xlabel('Panjang Tulang Daun (mm)')
            axes[1, 0].set_ylabel('Confidence')
            axes[1, 0].set_title('Confidence vs Panjang Tulang Daun')
            axes[1, 0].grid(True, alpha=0.3)
            # --- PERBAIKAN: Memperbarui scatter plot untuk rasio bentuk ---
            axes[1, 1].scatter(df['Rasio Bentuk Daun'], df['Confidence'], c=df['Confidence'], cmap='viridis', alpha=0.7, s=50)
            axes[1, 1].set_xlabel('Rasio Bentuk Daun')
            axes[1, 1].set_ylabel('Confidence')
            axes[1, 1].set_title('Confidence vs Rasio Bentuk Daun')
            axes[1, 1].grid(True, alpha=0.3)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            confidence_scatter_filename = f'grafik_confidence_scatter_{timestamp}.png'
            plt.savefig(confidence_scatter_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Grafik scatter confidence disimpan ke: {confidence_scatter_filename}")

            plt.figure(figsize=(10, 8))
            # --- PERBAIKAN: Memperbarui daftar kolom untuk heatmap ---
            numeric_cols = ['Panjang Daun (mm)', 'Lebar Daun (mm)', 'Keliling Daun (mm)',
                           'Panjang Tulang Daun (mm)', 'Rasio Bentuk Daun', 'Confidence']
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Heatmap Korelasi Confidence dengan Fitur Morfologi', fontsize=14, fontweight='bold')
            plt.tight_layout()
            correlation_heatmap_filename = f'grafik_korelasi_confidence_{timestamp}.png'
            plt.savefig(correlation_heatmap_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Grafik korelasi confidence disimpan ke: {correlation_heatmap_filename}")

            print("\n📊 Analisis perbandingan antar varietas selesai!")
            print("\n🎉 SEMUA PROSES SELESAI!")
            print(f"📄 File hasil: {csv_filename}")
            print(f"📊 Grafik perbandingan: {grafik_filename}")
            print(f"📊 Grafik rata-rata: {grafik_mean_filename}")
            print(f"📊 Scatter plot: {scatter_filename}")
            print(f"📊 Boxplot confidence: {confidence_boxplot_filename}")
            print(f"📊 Rata-rata confidence: {confidence_bar_filename}")
            print(f"📊 Scatter confidence: {confidence_scatter_filename}")
            print(f"📊 Korelasi confidence: {correlation_heatmap_filename}")
        else:
            print("\n❌ Tidak ada gambar berhasil dianalisis")
    else:
        print("❌ Gagal memuat model YOLO")
    print("\n" + "="*60)
    print("PROGRAM SELESAI")
    print("="*60)

if __name__ == "__main__":
    main()