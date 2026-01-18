import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# 1. MISMA CNN QUE EN EL TRAIN (DNNclass)
# =========================

class DNNclass(nn.Module):
    def __init__(self, num_classes):
        super(DNNclass, self).__init__()
        
        # Bloque 1
        # Entrada: 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) # -> 64 x 64
        
        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # pool -> 32 x 32
        
        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # pool -> 16 x 16
        
        # Bloque 4 (Añadido para dar más potencia)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # pool -> 8 x 8
        
        # CAPA FLATTEN + DENSE (Estilo clásico Kaggle)
        # La imagen final es 128 filtros x 8 x 8 píxeles
        self.flatten_size = 128 * 8 * 8 
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5) # Importante para no memorizar
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Bloque 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Bloque 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Bloque 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Bloque 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Aplanar (Flatten)
        x = x.view(x.size(0), -1) 
        
        # Capas Densas
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =========================
# 2. CONFIGURACIÓN DISPOSITIVO
# =========================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

MODEL_PATH = "models/best_model.pth"
DATA_DIR = "data/train"      # conjunto que quieres revisar
BATCH_SIZE = 32
SHOW_IMAGES = False          # pon True si quieres que se abran las imágenes
SAVE_MISCLASSIFIED = True
SAVE_DIR = "misclassified"   # carpeta donde se copiarán los errores

# =========================
# 3. CARGAR MODELO
# =========================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model = DNNclass(num_classes=2).to(device)
model.load_state_dict(checkpoint)
model.eval()

class_names = ['cats', 'dogs']
print("Classes:", class_names)

# =========================
# 4. DATASET Y DATALOADER (MISMAS TRANSFORMS)
# =========================

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Loaded {len(dataset)} images from {DATA_DIR}")

# =========================
# 5. RECORRER Y LOCALIZAR MAL CLASIFICADAS
# =========================

misclassified = []
global_index = 0   # índice sobre dataset.samples

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        batch_size_actual = labels.size(0)
        for i in range(batch_size_actual):
            true_label = labels[i].item()
            pred_label = preds[i].item()
            if true_label != pred_label:
                # Usamos el índice global para recuperar la ruta original
                img_path, _ = dataset.samples[global_index + i]
                misclassified.append((img_path, true_label, pred_label))

        global_index += batch_size_actual

# =========================
# 6. MOSTRAR / GUARDAR RESULTADOS
# =========================

print(f"\nTotal misclassified images: {len(misclassified)}\n")

if SAVE_MISCLASSIFIED:
    os.makedirs(SAVE_DIR, exist_ok=True)

for img_path, true_label, pred_label in misclassified:
    print(f"- {img_path} | true: {class_names[true_label]} | pred: {class_names[pred_label]}")

    if SAVE_MISCLASSIFIED:
        base = os.path.basename(img_path)
        dest_name = f"{class_names[true_label]}_as_{class_names[pred_label]}_{base}"
        dest_path = os.path.join(SAVE_DIR, dest_name)
        shutil.copy(img_path, dest_path)

    if SHOW_IMAGES:
        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)
        plt.title(f"True: {class_names[true_label]} | Pred: {class_names[pred_label]}")
        plt.axis("off")
        plt.show()

if SAVE_MISCLASSIFIED:
    print(f"\nMisclassified images copied to: {SAVE_DIR}")
