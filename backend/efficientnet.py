# Import library yang diperlukan
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import torchvision.models as models
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# Periksa status GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Bersihkan memori GPU
torch.cuda.empty_cache()

# Tetapkan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tetapkan jumlah kelas
num_classes = 13
print(f"Jumlah kelas: {num_classes}")

# Definisikan transformasi untuk data
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path dataset
dataset_path = '/kaggle/input/newdataset'
train_path = os.path.join(dataset_path, 'train')
valid_path = os.path.join(dataset_path, 'valid')
test_path = os.path.join(dataset_path, 'test')

# Definisikan dataset kustom untuk struktur dataset dengan _classes.csv one-hot
class ChiliDatasetCSV(Dataset):
    def __init__(self, data_dir, csv_path=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Baca file CSV jika ada
        if csv_path and os.path.exists(csv_path):
            self.classes_df = pd.read_csv(csv_path)
            
            # Dapatkan nama kelas dari kolom (kecuali kolom filename)
            self.classes = [col for col in self.classes_df.columns if col != 'filename']
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            
            # Buat mapping dari filename ke label
            self.file_to_label = {}
            for _, row in self.classes_df.iterrows():
                filename = row['filename']
                # Cari kelas dengan nilai 1 (one-hot encoding)
                for cls in self.classes:
                    if row[cls] == 1:
                        self.file_to_label[filename] = self.class_to_idx[cls]
                        break
        else:
            # Fallback jika tidak ada CSV
            self.classes = []
            self.class_to_idx = {}
            self.file_to_label = {}
            
            # Dapatkan semua file gambar
            image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Ekstrak label dari nama file
            for filename in image_files:
                # Asumsikan format nama file: label_filename.ext
                parts = filename.split('_', 1)
                if len(parts) > 1:
                    label = parts[0]
                    if label not in self.classes:
                        self.classes.append(label)
                        self.class_to_idx[label] = len(self.classes) - 1
                    self.file_to_label[filename] = self.class_to_idx[label]
            
            self.classes = sorted(self.classes)
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Dapatkan semua file gambar
        self.images = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Buat labels
        self.labels = []
        for img_name in self.images:
            if img_name in self.file_to_label:
                self.labels.append(self.file_to_label[img_name])
            else:
                # Jika tidak ada label, berikan label default
                self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Coba baca file _classes.csv
train_csv_path = os.path.join(dataset_path, 'train', '_classes.csv')
valid_csv_path = os.path.join(dataset_path, 'valid', '_classes.csv')
test_csv_path = os.path.join(dataset_path, 'test', '_classes.csv')

# Periksa apakah file CSV ada
print(f"\nFile CSV train ada: {os.path.exists(train_csv_path)}")
print(f"File CSV valid ada: {os.path.exists(valid_csv_path)}")
print(f"File CSV test ada: {os.path.exists(test_csv_path)}")

# Jika file CSV ada, tampilkan isinya
if os.path.exists(train_csv_path):
    print("\nIsi file _classes.csv untuk train:")
    train_df = pd.read_csv(train_csv_path)
    print(train_df.head())
    
    # Dapatkan nama kelas dari kolom
    classes = [col for col in train_df.columns if col != 'filename']
    print(f"\nJumlah kelas unik: {len(classes)}")
    print(f"Kelas: {classes}")

# Buat dataset
train_dataset = ChiliDatasetCSV(train_path, train_csv_path, transform=train_transform)
val_dataset = ChiliDatasetCSV(valid_path, valid_csv_path, transform=val_transform)
test_dataset = ChiliDatasetCSV(test_path, test_csv_path, transform=val_transform)

print(f"\nJumlah data training: {len(train_dataset)}")
print(f"Jumlah data validasi: {len(val_dataset)}")
print(f"Jumlah data test: {len(test_dataset)}")
print(f"Jumlah kelas: {len(train_dataset.classes)}")
print(f"Kelas: {train_dataset.classes}")

# Buat dataloader
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Gunakan EfficientNet dari torchvision dengan beberapa modifikasi untuk akurasi lebih tinggi
print("\nMenginisialisasi EfficientNet-B0 dengan modifikasi...")
try:
    model = models.efficientnet_b0(pretrained=True)
    
    # Ganti classifier dengan yang lebih kompleks untuk akurasi lebih tinggi
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Unfreeze beberapa layer terakhir untuk fine-tuning
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    
    model = model.to(device)
    print("EfficientNet-B0 berhasil diinisialisasi dengan modifikasi!")
    
    # Test model
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_output = model(dummy_input)
    print(f"Test berhasil! Output shape: {dummy_output.shape}")
    
except Exception as e:
    print(f"Error dengan EfficientNet: {e}")
    raise

# Gunakan optimizer yang lebih baik
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Atur learning rate scheduler yang lebih agresif
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=2,  # Lebih sabar
    verbose=True
)

# Loss function dengan label smoothing untuk generalisasi lebih baik
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

# GradScaler untuk mixed precision training
if device.type == 'cuda':
    from torch.amp import GradScaler
    scaler = GradScaler('cuda')
    print("Menggunakan mixed precision training")
else:
    scaler = None
    print("Tidak menggunakan mixed precision training")

# Fungsi untuk mendapatkan learning rate saat ini
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# List untuk menyimpan history
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
best_val_acc = 0.0

# Path untuk menyimpan model terbaik
best_model_path = "best_efficientnet_model.pth"

# Fungsi training dengan augmentasi tambahan
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        try:
            # Pastikan input dan label dalam format yang benar
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            
            # Pindahkan ke device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Debugging: cetak shape dan rentang nilai (hanya untuk batch pertama)
            if batch_idx == 0:
                print(f"Input shape: {inputs.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Label min/max: {labels.min().item()}/{labels.max().item()}")
                print(f"Number of classes in batch: {len(torch.unique(labels))}")
            
            # Pastikan label dalam rentang yang valid
            assert labels.min() >= 0, f"Label negatif ditemukan: {labels.min()}"
            assert labels.max() < num_classes, f"Label melebihi jumlah kelas: {labels.max()} (harus < {num_classes})"
            
            optimizer.zero_grad()
            
            if scaler is not None and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        except Exception as e:
            print(f"Error pada batch {batch_idx}: {e}")
            continue
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct.double() / total
    
    return epoch_loss, epoch_acc

# Fungsi validasi dengan metrik lengkap
def validate(model, dataloader, criterion, device, class_names=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
            try:
                # Pastikan input dan label dalam format yang benar
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Pastikan label dalam rentang yang valid
                assert labels.min() >= 0, f"Label negatif ditemukan: {labels.min()}"
                assert labels.max() < num_classes, f"Label melebihi jumlah kelas: {labels.max()} (harus < {num_classes})"
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)
                
                # Simpan prediksi dan label untuk metrik lengkap
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"Error during validation: {e}")
                continue
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct.double() / total
    
    # Hitung metrik lengkap jika diminta
    metrics = {}
    if class_names is not None:
        # Precision, recall, F1-score per kelas
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Metrik macro
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Metrik weighted
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
    
    return epoch_loss, epoch_acc.item(), metrics

# Debugging dataset dan dataloader
print("\nMemeriksa dataset dan dataloader...")
print(f"Jumlah data training: {len(train_loader.dataset)}")
print(f"Jumlah data validasi: {len(valid_loader.dataset)}")

# Periksa beberapa sampel dari dataloader
sample_inputs, sample_labels = next(iter(train_loader))
print(f"Shape sample inputs: {sample_inputs.shape}")
print(f"Shape sample labels: {sample_labels.shape}")
print(f"Sample labels: {sample_labels[:10]}")
print(f"Unique labels: {torch.unique(sample_labels)}")
print(f"Label range: {sample_labels.min()} - {sample_labels.max()}")

# Pastikan model dapat menerima input dengan ukuran ini
try:
    with torch.no_grad():
        sample_output = model(sample_inputs.to(device))
    print(f"Model output shape: {sample_output.shape}")
    print("Model test berhasil!")
except Exception as e:
    print(f"Error saat test model: {e}")
    raise

# Training loop dengan early stopping
NUM_EPOCHS = 20
patience = 5
patience_counter = 0
best_val_f1 = 0.0  # Ubah ke F1-score untuk kriteria terbaik

try:
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)
        
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc.item())
        
        # Validation dengan metrik lengkap
        val_loss, val_acc, val_metrics = validate(model, valid_loader, criterion, device, train_dataset.classes)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # Update scheduler berdasarkan validation loss
        scheduler.step(val_loss)
        
        # Dapatkan dan cetak learning rate saat ini
        current_lr = get_lr(optimizer)
        print(f'Current learning rate: {current_lr:.2e}')
        
        # Print hasil
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Val Precision (macro): {val_metrics["precision_macro"]:.4f}')
        print(f'Val Recall (macro): {val_metrics["recall_macro"]:.4f}')
        print(f'Val F1-Score (macro): {val_metrics["f1_macro"]:.4f}')
        
        # Simpan model terbaik berdasarkan F1-score macro
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_val_acc = val_acc  # Simpan akurasi juga untuk referensi
            torch.save(model.state_dict(), best_model_path)
            print(f"Model terbaik disimpan! F1-Score: {best_val_f1:.4f}")
            patience_counter = 0
            # Simpan metrik terbaik untuk evaluasi nanti
            best_val_metrics = val_metrics
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered setelah {epoch+1} epoch")
                break

except Exception as e:
    print(f"Error during training: {e}")
    # Simpan model terakhir meskipun ada error
    torch.save(model.state_dict(), "emergency_save_model.pth")
    print("Model disimpan secara darurat karena error")

print(f"\nTraining selesai! F1-Score validasi terbaik: {best_val_f1:.4f}")

# Evaluasi model terbaik di test set
print("\n" + "="*50)
print("Evaluasi Model Terbaik di Test Set")
print("="*50)

# Load model terbaik
model.load_state_dict(torch.load(best_model_path))

# Evaluasi di test set
test_loss, test_acc, test_metrics = validate(model, test_loader, criterion, device, train_dataset.classes)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision (macro): {test_metrics['precision_macro']:.4f}")
print(f"Test Recall (macro): {test_metrics['recall_macro']:.4f}")
print(f"Test F1-Score (macro): {test_metrics['f1_macro']:.4f}")
print(f"Test Precision (weighted): {test_metrics['precision_weighted']:.4f}")
print(f"Test Recall (weighted): {test_metrics['recall_weighted']:.4f}")
print(f"Test F1-Score (weighted): {test_metrics['f1_weighted']:.4f}")

# Tampilkan classification report
print("\nClassification Report (Test Set):")
print(classification_report(
    [train_dataset.classes[i] for i in test_metrics['confusion_matrix'].sum(axis=1).argsort()],
    [train_dataset.classes[i] for i in test_metrics['confusion_matrix'].sum(axis=0).argsort()],
    target_names=train_dataset.classes
))

# Visualisasi confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    test_metrics['confusion_matrix'], 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes
)
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Visualisasi metrik per kelas
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# Precision per kelas
sns.barplot(x=train_dataset.classes, y=test_metrics['precision_per_class'], ax=ax1)
ax1.set_title('Precision per Class')
ax1.set_ylim(0, 1)
ax1.set_xticklabels(train_dataset.classes, rotation=45, ha='right')

# Recall per kelas
sns.barplot(x=train_dataset.classes, y=test_metrics['recall_per_class'], ax=ax2)
ax2.set_title('Recall per Class')
ax2.set_ylim(0, 1)
ax2.set_xticklabels(train_dataset.classes, rotation=45, ha='right')

# F1-score per kelas
sns.barplot(x=train_dataset.classes, y=test_metrics['f1_per_class'], ax=ax3)
ax3.set_title('F1-Score per Class')
ax3.set_ylim(0, 1)
ax3.set_xticklabels(train_dataset.classes, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Analisis kesalahan
print("\nAnalisis Kesalahan (Test Set):")
error_analysis = []
for true_label in range(len(train_dataset.classes)):
    for pred_label in range(len(train_dataset.classes)):
        if true_label != pred_label:
            error_count = test_metrics['confusion_matrix'][true_label, pred_label]
            if error_count > 0:
                error_analysis.append({
                    'true_class': train_dataset.classes[true_label],
                    'pred_class': train_dataset.classes[pred_label],
                    'count': error_count
                })

# Urutkan berdasarkan jumlah kesalahan
error_analysis_df = pd.DataFrame(error_analysis)
if not error_analysis_df.empty:
    error_analysis_df = error_analysis_df.sort_values('count', ascending=False)
    print("\nTop 5 kesalahan klasifikasi:")
    print(error_analysis_df.head(5))
    
    # Analisis per kelas yang sering salah
    print("\nKelas yang paling sering salah klasifikasi:")
    true_errors = test_metrics['confusion_matrix'].sum(axis=1) - np.diag(test_metrics['confusion_matrix'])
    error_by_class = pd.DataFrame({
        'class': train_dataset.classes,
        'error_count': true_errors,
        'total_samples': test_metrics['support_per_class'],
        'error_rate': true_errors / test_metrics['support_per_class']
    }).sort_values('error_count', ascending=False)
    print(error_by_class)